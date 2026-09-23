import simpy
from sim_tools.distributions import Poisson
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from tqdm import tqdm

# NEW IMPORTS
from vidigi.logging import EventLogger, TrialLogger
from vidigi.utils import create_event_position_df, EventPosition
from vidigi.prep import reshape_for_animations, generate_animation_df
from vidigi.animation import generate_animation
from vidigi.process_mapping import (
    add_sim_timestamp,
    discover_dfg,
    dfg_to_graphviz,
    dfg_to_cytoscape,
)
from IPython.display import display
# END NEW IMPORTS

os.chdir(Path(__file__).parent)


class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_first_apt = pd.NA
        self.q_time_fu_apts = {}
        self.current_apt_id = 0


class Param:
    def __init__(
        self,
        num_slots_per_day=20,
        mean_referrals_per_day=6.0,
        prob_next_apt_dict={
            0: 0.7,
            1: 0.9,
            2: 0.85,
            3: 0.75,
            4: 0.6,
            5: 0.5,
            6: 0.4,
            7: 0.35,
            8: 0.15,
            9: 0.1,
            10: 0.05,
            11: 0.01,
            12: 0.001,
        },
        gap_between_fu_apts=90,
        results_collection_period=(365 * 5),
        warm_up_period=365,
        num_replications=100,
        num_replications_warm_up_assessment=5,
        warm_up_assessment_sim_length_scaler=5,
        cumulative_mean_tracker_interval=500,
    ):
        self.num_slots_per_day = num_slots_per_day
        self.mean_referrals_per_day = mean_referrals_per_day
        self.prob_next_apt_dict = prob_next_apt_dict
        self.max_key_prob_next_apt_dict = max(prob_next_apt_dict)
        self.gap_between_fu_apts = gap_between_fu_apts
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.sim_duration = warm_up_period + results_collection_period
        self.num_replications = num_replications
        self.num_replications_warm_up_assessment = num_replications_warm_up_assessment
        self.warm_up_assessment_sim_length_scaler = warm_up_assessment_sim_length_scaler
        self.cumulative_mean_tracker_interval = cumulative_mean_tracker_interval


class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.daily_slots = simpy.Container(
            self.env, self.param.num_slots_per_day, init=self.param.num_slots_per_day
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(2)

        self.referrals_per_day_dist = Poisson(
            rate=self.param.mean_referrals_per_day, random_seed=seeds[0]
        )

        self.follow_up_decider_rng = np.random.default_rng(seeds[1])

        self.list_of_patients = []
        self.mean_q_time_first_apt = pd.NA
        self.sd_q_time_first_apt = pd.NA
        self.perc_90_q_time_first_apt = pd.NA
        self.mean_q_time_fu_apts = {}
        self.sd_q_time_fu_apts = {}
        self.perc_90_q_time_fu_apts = {}

        self.logger = EventLogger(env=self.env, run_number=self.replication_id)  # NEW

    def generator_new_referrals(self):
        while True:
            todays_referrals = self.referrals_per_day_dist.sample()
            # print (f"Day {self.env.now}: {todays_referrals} referrals")

            for referral in range(todays_referrals):
                self.patient_counter += 1
                p = Patient(self.patient_counter)
                self.list_of_patients.append(p)
                self.env.process(self.appointment_governor(p))

            yield self.env.timeout(1)

    def attend_first_apt(self, patient):
        start_q_first_apt = self.env.now
        self.logger.log_queue(
            entity_id=patient.id,
            event="start_queue_first_appt",
            visit_number=patient.current_apt_id,
        )

        yield self.daily_slots.get(1)

        # print (f"Patient {patient.id} attending FIRST APPOINTMENT")

        end_q_first_apt = self.env.now
        self.logger.log_queue(
            entity_id=patient.id,
            event="have_first_appt",
            visit_number=patient.current_apt_id,
        )  # NEW

        if self.env.now > self.param.warm_up_period:
            patient.q_time_first_apt = end_q_first_apt - start_q_first_apt

        yield self.env.timeout(1)

        self.logger.log_queue(
            entity_id=patient.id,
            event=f"finish_fu_appt_{patient.current_apt_id}",
            visit_number=patient.current_apt_id,
        )

        yield self.daily_slots.put(1)

    def delay_until_apt_due(self, patient):
        yield self.env.timeout(self.param.gap_between_fu_apts)

    def attend_fu_apt(self, patient):
        start_q_fu_apt = self.env.now
        self.logger.log_queue(
            entity_id=patient.id,
            event=f"start_queue_fu_appt_{patient.current_apt_id}",
            visit_number=patient.current_apt_id,
        )

        yield self.daily_slots.get(1)

        # print (
        #    f"Patient {patient.id} attending FU appointment",
        #    patient.current_apt_id
        # )

        end_q_fu_apt = self.env.now
        self.logger.log_queue(
            entity_id=patient.id,
            event=f"have_fu_appt_{patient.current_apt_id}",
            visit_number=patient.current_apt_id,
        )

        if self.env.now > self.param.warm_up_period:
            patient.q_time_fu_apts[patient.current_apt_id] = (
                end_q_fu_apt - start_q_fu_apt
            )

        yield self.env.timeout(1)

        self.logger.log_queue(
            entity_id=patient.id,
            event=f"finish_fu_appt_{patient.current_apt_id}",
            visit_number=patient.current_apt_id,
        )

        yield self.daily_slots.put(1)

    def appointment_governor(self, patient):
        self.logger.log_arrival(entity_id=patient.id)
        yield self.env.process(self.attend_first_apt(patient))

        while True:
            apt_id_clamp = min(
                patient.current_apt_id, self.param.max_key_prob_next_apt_dict
            )

            if (
                self.follow_up_decider_rng.random()
                < self.param.prob_next_apt_dict[apt_id_clamp]
            ):
                patient.current_apt_id += 1
                yield self.env.process(self.delay_until_apt_due(patient))
                yield self.env.process(self.attend_fu_apt(patient))
            else:
                self.logger.log_departure(entity_id=patient.id)
                return

    def cumulative_mean_tracker(self):
        yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

        self.cumulative_mean_df = pd.DataFrame(columns=["Simulation Time"])

        while True:
            if self.list_of_patients:
                df_patients = pd.DataFrame([vars(p) for p in self.list_of_patients])

                df_patients = df_patients.join(
                    df_patients["q_time_fu_apts"].apply(pd.Series)
                )
                df_patients.drop(columns=["q_time_fu_apts"], inplace=True)

                row_of_means = df_patients.mean()
                row_of_means["Simulation Time"] = self.env.now
            else:
                row_of_means = pd.Series({"Simulation Time": self.env.now})

            self.cumulative_mean_df = pd.concat(
                [self.cumulative_mean_df, row_of_means.to_frame().T], ignore_index=True
            )

            yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

    def run_model(self):
        self.env.process(self.generator_new_referrals())
        self.env.run(until=self.param.sim_duration)

    def run_warm_up_assessment(self):
        old_warm_up = self.param.warm_up_period
        self.param.warm_up_period = 0
        self.param.sim_duration_warm_up_assessment = (
            self.param.results_collection_period
            * self.param.warm_up_assessment_sim_length_scaler
        )
        self.env.process(self.generator_new_referrals())
        self.env.process(self.cumulative_mean_tracker())
        self.env.run(until=self.param.sim_duration_warm_up_assessment)
        self.param.warm_up_period = old_warm_up

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dataframe = pd.DataFrame(entity.__dict__ for entity in entity_list)

        entity_dataframe = entity_dataframe.join(
            entity_dataframe["q_time_fu_apts"].apply(pd.Series)
        )
        entity_dataframe.drop(columns=["q_time_fu_apts"], inplace=True)

        return entity_dataframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_first_apt = entity_dataframe["q_time_first_apt"].mean()
        self.sd_q_time_first_apt = entity_dataframe["q_time_first_apt"].std()
        self.perc_90_q_time_first_apt = entity_dataframe["q_time_first_apt"].quantile(
            0.9
        )

        for fu_num in sorted(
            col for col in entity_dataframe.columns if isinstance(col, int)
        ):
            self.mean_q_time_fu_apts[fu_num] = entity_dataframe[fu_num].mean()
            self.sd_q_time_fu_apts[fu_num] = entity_dataframe[fu_num].std()
            self.perc_90_q_time_fu_apts[fu_num] = entity_dataframe[fu_num].quantile(0.9)

    # NEW
    def get_vidigi_event_log(self):
        return self.logger.to_dataframe()


class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_first_apt = pd.NA
        self.trial_sd_q_time_first_apt = pd.NA
        self.trial_perc_90_q_time_first_apt = pd.NA
        self.ci_lower_q_time_first_apt = pd.NA
        self.ci_upper_q_time_first_apt = pd.NA
        self.se_q_time_first_apt = pd.NA
        self.trial_mean_q_time_fu_apts = {}
        self.trial_sd_q_time_fu_apts = {}
        self.trial_perc_90_q_time_fu_apts = {}
        self.ci_lower_q_time_fu_apts = {}
        self.ci_upper_q_time_fu_apts = {}
        self.se_q_time_fu_apts = {}
        self.warm_up_trial = False
        self.trial_logger = TrialLogger()  # NEW

    def run_trial(self):
        for replication_id in tqdm(
            range(self.param.num_replications), desc="Running Trial", unit="replication"
        ):
            model_replication = Model(self.param, replication_id)
            model_replication.run_model()
            patient_df = model_replication.convert_entity_list_to_dataframe(
                model_replication.list_of_patients
            )
            model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(model_replication)
            self.trial_logger.add_log(model_replication.logger)  # NEW

    def run_warm_up_assessment_trial(self):
        self.warm_up_trial = True
        self.list_of_cumulative_mean_dfs = []

        for wu_replication_id in tqdm(
            range(self.param.num_replications_warm_up_assessment),
            desc="Running Warm Up Assessment",
            unit="replication",
        ):
            wu_model_replication = Model(self.param, wu_replication_id)
            wu_model_replication.run_warm_up_assessment()
            patient_df = wu_model_replication.convert_entity_list_to_dataframe(
                wu_model_replication.list_of_patients
            )
            wu_model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(wu_model_replication)

            self.list_of_cumulative_mean_dfs.append(
                wu_model_replication.cumulative_mean_df
            )

        x_col = "Simulation Time"
        y_cols = []

        for df in self.list_of_cumulative_mean_dfs:
            for col in df.columns:
                if (
                    col not in ["Simulation Time", "id", "current_apt_id"]
                    and col not in y_cols
                ):
                    y_cols.append(col)

        for col in y_cols:
            fig = go.Figure()

            for i, df in enumerate(self.list_of_cumulative_mean_dfs, start=1):
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df.get(col, pd.Series(np.nan, index=df.index)),
                        mode="lines",
                        name=f"replication_{i}",
                        line=dict(color="lightblue", width=1),
                    )
                )

            combined = []

            for df in self.list_of_cumulative_mean_dfs:
                combined.append(df.reindex(columns=[x_col, col]))

            df_all_reps = pd.concat(combined)

            df_all_reps = df_all_reps.sort_values(x_col)

            mean_across_reps_df = df_all_reps.groupby(x_col, as_index=False)[col].mean()

            mean_across_reps_df["overall_cumulative"] = (
                mean_across_reps_df[col].expanding().mean()
            )

            fig.add_trace(
                go.Scatter(
                    x=mean_across_reps_df[x_col],
                    y=mean_across_reps_df["overall_cumulative"],
                    mode="lines",
                    name="overall_mean",
                    line=dict(color="darkblue", width=4),
                )
            )

            fig.update_layout(
                title=f"Cumulative Mean - {col}",
                xaxis_title=x_col,
                yaxis_title="Cumulativve Mean",
            )

            fig.show()
            fig.write_html(f"govern_cumul_mean_{col}.html")

    def calculate_trial_results(self):
        if self.warm_up_trial:
            total_reps = self.param.num_replications_warm_up_assessment
        else:
            total_reps = self.param.num_replications

        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in self.list_of_simulation_replications
        )

        self.trial_mean_q_time_first_apt = self.replication_df[
            "mean_q_time_first_apt"
        ].mean()

        self.trial_sd_q_time_first_apt = self.replication_df[
            "mean_q_time_first_apt"
        ].std()

        self.trial_perc_90_q_time_first_apt = self.replication_df[
            "mean_q_time_first_apt"
        ].quantile(0.9)

        self.se_q_time_first_apt = self.trial_sd_q_time_first_apt / math.sqrt(
            total_reps
        )

        t = stats.t.ppf(0.975, df=total_reps - 1)

        self.ci_lower_q_time_first_apt = self.trial_mean_q_time_first_apt - (
            t * self.se_q_time_first_apt
        )

        self.ci_upper_q_time_first_apt = self.trial_mean_q_time_first_apt + (
            t * self.se_q_time_first_apt
        )

        fu_means = pd.DataFrame(
            [
                replication.mean_q_time_fu_apts
                for replication in self.list_of_simulation_replications
            ]
        )

        self.trial_mean_q_time_fu_apts = fu_means.mean().to_dict()
        self.trial_sd_q_time_fu_apts = fu_means.std().to_dict()
        self.trial_perc_90_q_time_fu_apts = fu_means.quantile(0.9).to_dict()

        fu_means_means = fu_means.mean()
        fu_sd_values = fu_means.std()
        n_fu_means = fu_means.count()
        fu_se_values = fu_sd_values / np.sqrt(n_fu_means)
        self.se_q_time_fu_apts = fu_se_values.to_dict()

        t = pd.Series(stats.t.ppf(0.975, df=n_fu_means - 1), index=n_fu_means.index)
        self.trial_ci_lower_q_time_fu_apts = (
            fu_means_means - t * fu_se_values
        ).to_dict()
        self.trial_ci_upper_q_time_fu_apts = (
            fu_means_means + t * fu_se_values
        ).to_dict()


class Animation:
    def __init__(self, event_log, params):
        self.event_log = event_log
        self.params = params

        # The governor model logs a "start_queue_..."/"have_..." pair for the
        # first appointment, then repeats that same pair - suffixed with the
        # visit number (patient.current_apt_id) - for every follow-up
        # appointment. There's no hard cap on how many follow-up appointments
        # a patient can have (prob_next_apt_dict keeps a non-zero probability
        # of another visit forever), so we generate rows for a generous range
        # of visit numbers beyond the highest key in prob_next_apt_dict rather
        # than hardcoding an exact list.
        #
        # Layout: one row per appointment (first appointment, then each
        # follow-up visit), with the "waiting" queue on the left ~65% of the
        # canvas and the "attending" spot on the right ~35% - see
        # vidigi's example_17_resourceless_larger_queues, which also shows
        # that with queues this large you need to cap step_snapshot_max low
        # and lean on step_snapshot_limit_gauges rather than trying to fit
        # every waiting patient on screen.
        self.canvas_width = 1400
        self.canvas_height = 1400
        waiting_x = 350
        attending_x = 950
        row_height = 100
        first_row_y = 1200

        # Set up the fixed event positions first
        event_positions = [
            EventPosition(
                event="arrival", x=0, y=int(first_row_y + row_height), label="Entrance"
            ),
            EventPosition(
                event="start_queue_first_appt",
                x=int(waiting_x),
                y=int(first_row_y),
                label="Waiting for First Appointment",
            ),
            EventPosition(
                event="have_first_appt",
                x=int(attending_x),
                y=int(first_row_y),
                label="Attending First Appointment",
            ),
        ]

        # Find out the maximum number of follow-ups anyone had
        max_fu_visit_number = int(self.event_log["visit_number"].max())
        # print(max_fu_visit_number)

        # Generate the max number of lines required
        for visit_number in range(1, max_fu_visit_number + 1):
            row_y = first_row_y - visit_number * row_height

            event_positions.append(
                EventPosition(
                    event=f"start_queue_fu_appt_{visit_number}",
                    x=int(waiting_x),
                    y=int(row_y),
                    label=f"Waiting for Follow-Up Appointment {visit_number}",
                )
            )
            event_positions.append(
                EventPosition(
                    event=f"have_fu_appt_{visit_number}",
                    x=int(attending_x),
                    y=int(row_y),
                    label=f"Attending Follow-Up Appointment {visit_number}",
                )
            )

        event_positions.append(
            EventPosition(event="depart", x=900, y=100, label="Exit")
        )

        self.layout = create_event_position_df(event_positions)

    def build_animation(self, time_interval=1):
        # Aggressively cap how many individual icons are drawn per queue -
        # with queues this large, showing every waiting patient just produces
        # unreadable stacks of icons and overlapping "+ N more" text.
        step_snapshot_max = 20

        reshaped_df = reshape_for_animations(
            event_log=self.event_log,
            every_x_time_units=time_interval,
            limit_duration=self.params.sim_duration,
            step_snapshot_max=step_snapshot_max,
        )

        # Filter to account for the warm-up
        reshaped_df = reshaped_df[
            (reshaped_df["snapshot_time"] > self.params.warm_up_period)
        ]

        animation_df = generate_animation_df(
            full_entity_df=reshaped_df,
            event_position_df=self.layout,
            step_snapshot_max=step_snapshot_max,
            # Keep each queue on a single row (matches step_snapshot_max, so
            # no in-row wrapping) rather than spilling into extra sub-rows.
            wrap_queues_at=20,
            gap_between_entities=15,
            gap_between_queue_rows=25,
            # Swap the '+ N more' overflow text for a gauge once a queue
            # exceeds step_snapshot_max, instead of dense unreadable text.
            step_snapshot_limit_gauges=True,
        )

        return generate_animation(
            full_entity_df_plus_pos=animation_df,
            event_position_df=self.layout,
            simulation_time_unit="days",  # NEW
            entity_icon_size=8,
            text_size=12,
            plotly_width=900,
            plotly_height=900,
            override_x_max=self.canvas_width,
            override_y_max=self.canvas_height,
            frame_transition_duration=0,
            time_display_units="simulation_day_clock",
        )


class ProcessMap:
    def __init__(self, event_log, params):
        self.event_log = event_log
        self.params = params
        self.time_unit = "days"  # NEW

    def build_process_map(self, interactive=True):
        filtered_event_log = self.event_log.copy()

        # Account for warm-up
        filtered_event_log = filtered_event_log[
            filtered_event_log["time"] > self.params.warm_up_period
        ]

        # NEW
        # Let's filter to just the 'have' events
        filtered_event_log = filtered_event_log[
            filtered_event_log["event"].str.contains("have|arrival|depart")
        ]

        filtered_event_log_timestamp = add_sim_timestamp(
            filtered_event_log,
            time_unit=self.time_unit,  # UPDATED
            sim_start="09:00:00",
        )

        nodes, edges = discover_dfg(
            filtered_event_log_timestamp,
            case_col="entity_id",
            time_unit=self.time_unit,  # NEW
        )

        if interactive:
            cytoscape_widget = dfg_to_cytoscape(
                nodes,
                edges,
                min_frequency=5,
                layout_name="dagre",
                layout_orientation="TD",  # UPDATED
                spacing_factor=2,
                width=1400,
                time_unit=self.time_unit,  # NEW
            )
            display(cytoscape_widget)

        else:
            graphviz_graph = dfg_to_graphviz(
                nodes,
                edges,
                min_frequency=5,
                direction="TD",  # NEW
                time_unit=self.time_unit,  # NEW
            )
            display(graphviz_graph)


# BASE CASE PARAMETERS DEFINITION
base_case_params = Param(num_replications=3)

# WARM UP ASSESSMENT
# warm_up_assessment_trial = Trial(base_case_params)
# warm_up_assessment_trial.run_warm_up_assessment_trial()
# warm_up_assessment_trial.calculate_trial_results()

# BASE CASE TRIAL
base_case_trial = Trial(base_case_params)
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()

print("BASE CASE TRIAL RESULTS")
print("-----------------------")
print("Queuing Time for First Appointment")
print(f"Mean: {base_case_trial.trial_mean_q_time_first_apt:.2f} days")
print(f"SD: {base_case_trial.trial_sd_q_time_first_apt:.2f} days")
print(f"90th Perc: {base_case_trial.trial_perc_90_q_time_first_apt:.2f} days")
print(f"SE: {base_case_trial.se_q_time_first_apt:.2f} days")
print(
    f"95% CI : ({base_case_trial.ci_lower_q_time_first_apt:.2f}, ",
    f"{base_case_trial.ci_upper_q_time_first_apt:.2f}) days",
)

print()

print("Queuing Time for Follow-Up Appointments (beyond due date)")

for fu_num in sorted(base_case_trial.trial_mean_q_time_fu_apts):
    print(
        f"FU {fu_num} :",
        f"Mean : {base_case_trial.trial_mean_q_time_fu_apts[fu_num]:.2f}",
        f"SD : {base_case_trial.trial_sd_q_time_fu_apts[fu_num]:.2f}",
        f"90th P : {base_case_trial.trial_perc_90_q_time_fu_apts[fu_num]:.2f}",
        f"SE : {base_case_trial.se_q_time_fu_apts[fu_num]:.2f}",
        f"95% CI :",
        f"({base_case_trial.trial_ci_lower_q_time_fu_apts[fu_num]:.2f},",
        f"{base_case_trial.trial_ci_upper_q_time_fu_apts[fu_num]:.2f}) days",
    )

base_case_event_log = base_case_trial.trial_logger.get_log_by_run(run=0, as_df=True)
print(base_case_event_log.head(10))

my_process_map = ProcessMap(base_case_event_log, base_case_params)
my_process_map.build_process_map(interactive=False)

my_animation = Animation(base_case_event_log, base_case_params)
# Because we're running it for longer, let's do a frame every two
# minutes to keep it generating quickly
fig = my_animation.build_animation(time_interval=2)  # UPDATED
fig.show()
