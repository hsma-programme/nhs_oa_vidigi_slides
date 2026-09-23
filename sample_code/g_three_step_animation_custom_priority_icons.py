"""
This is the one-step model from the second simpy session (2B)
"""

import simpy
from sim_tools.distributions import Lognormal
from sim_tools.time_dependent import NSPPThinning
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

from vidigi.logging import EventLogger, TrialLogger
from vidigi.utils import create_event_position_df, EventPosition
from vidigi.prep import reshape_for_animations, generate_animation_df
from vidigi.animation import generate_animation
from vidigi.resources import VidigiPriorityStore  # UPDATED
from vidigi.process_mapping import (
    add_sim_timestamp,
    discover_dfg,
    dfg_to_graphviz,
    dfg_to_cytoscape,
)
from IPython.display import display

os.chdir(Path(__file__).parent)


class Patient:
    def __init__(self, p_id, priority):
        self.id = p_id
        self.q_time_nurse = pd.NA
        self.arrival_time = pd.NA
        self.priority = priority


class Param:
    def __init__(
        self,
        patient_iat_csv,
        mean_nurse_consult_time=12,
        sd_nurse_consult_time=1,
        num_nurses=3,
        results_collection_period=480,
        warm_up_period=1500,
        num_replications=100,
        num_replications_warm_up_assessment=50,
        warm_up_asessment_sim_length_scaler=20,
        cumulative_mean_tracker_interval=5,
        nurse_unav_time=60,
        nurse_unav_freq=120,
        num_nurses_unav=2,
    ):
        self.mean_nurse_consult_time = mean_nurse_consult_time
        self.sd_nurse_consult_time = sd_nurse_consult_time
        self.num_nurses = num_nurses
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.sim_duration = warm_up_period + results_collection_period
        self.num_replications = num_replications

        self.num_replications_warm_up_assessment = num_replications_warm_up_assessment

        self.warm_up_assessment_sim_length_scaler = warm_up_asessment_sim_length_scaler

        self.cumulative_mean_tracker_interval = cumulative_mean_tracker_interval

        self.pt_arrivals_time_dependent_df = pd.read_csv(patient_iat_csv)

        self.nurse_unav_time = nurse_unav_time
        self.nurse_unav_freq = nurse_unav_freq
        self.num_nurses_unav = num_nurses_unav


class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.logger = EventLogger(env=self.env, run_number=self.replication_id)

        # UPDATED
        self.nurse = VidigiPriorityStore(
            self.env,
            num_resources=self.param.num_nurses,
            label="nurse",
            logger=self.logger,
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(4)  # NEW - added extra seed spawn

        self.patient_inter_dist = NSPPThinning(
            data=param.pt_arrivals_time_dependent_df,
            random_seed1=seeds[0],
            random_seed2=seeds[2],
        )
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time,
            random_seed=seeds[1],
        )
        # NEW
        self.patient_priority_rng = np.random.default_rng(seeds[3])

        self.list_of_patients = []
        self.mean_q_time_nurse = pd.NA
        self.sd_q_time_nurse = pd.NA
        self.perc_90_q_time_nurse = pd.NA

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            pat_pri_ran_gen = self.patient_priority_rng.random()

            if pat_pri_ran_gen < 0.2:
                patient_priority = 1
            elif pat_pri_ran_gen < 0.7:
                patient_priority = 2
            else:
                patient_priority = 3

            p = Patient(self.patient_counter, patient_priority)  # NEW
            self.list_of_patients.append(p)
            self.env.process(self.attend_clinic(p))

            sampled_inter = self.patient_inter_dist.sample(simulation_time=self.env.now)

            yield self.env.timeout(sampled_inter)

    def obstruct_nurse(self):
        next_departure_time = self.param.nurse_unav_freq

        while True:
            yield self.env.timeout(next_departure_time - self.env.now)

            time_should_go = next_departure_time
            time_to_return = time_should_go + self.param.nurse_unav_time

            # print(
            #     f"{self.param.num_nurses_unav} nurses should go at",
            #     f"{time_should_go:.2f}",
            # )

            for removal_candidate in range(self.param.num_nurses_unav):
                self.env.process(self.remove_one_nurse(time_to_return))

            next_departure_time += (
                self.param.nurse_unav_time + self.param.nurse_unav_freq
            )

    def remove_one_nurse(self, time_to_return):
        req = self.nurse.request(priority=-1)
        yield req
        time_went = self.env.now
        print(f"A nurse went at {time_went:.2f}")

        time_away = time_to_return - time_went
        yield self.env.timeout(time_away)
        self.nurse.release(req)
        print(f"A nurse returned at {self.env.now:.2f}")

    def cumulative_mean_tracker(self):
        yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

        self.cumulative_mean_df = pd.DataFrame(columns=["Simulation Time"])

        while True:
            if self.list_of_patients:
                df_patients = pd.DataFrame([vars(p) for p in self.list_of_patients])

                row_of_means = df_patients.mean()
                row_of_means["Simulation Time"] = self.env.now
            else:
                row_of_means = pd.Series({"Simulation Time": self.env.now})

            self.cumulative_mean_df = pd.concat(
                [self.cumulative_mean_df, row_of_means.to_frame().T], ignore_index=True
            )

            yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

    def attend_clinic(self, patient):
        patient.arrival_time = self.env.now
        # UPDATED
        # Note that we have added priority to our arrival log. We will do this
        # for every logged activity throughout the patient's journey.
        # All of the logging functions can take any number of additional fields
        # as additional arguments
        self.logger.log_arrival(
            entity_id=patient.id,
            patient_priority=patient.priority,  # NEW
        )

        start_q_nurse = self.env.now

        self.logger.log_queue(
            entity_id=patient.id,
            event="nurse_wait_begins",
            patient_priority=patient.priority,  # NEW
        )

        with self.nurse.request(
            priority=patient.priority,  # NEW
            entity_id=patient.id,
            start_event="being_seen_by_nurse",
            end_event="nurse_treatment_ends",
            patient_priority=patient.priority,  # NEW
        ) as req:
            yield req

            end_q_nurse = self.env.now
            patient.q_time_nurse = end_q_nurse - start_q_nurse

            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)

        self.logger.log_departure(
            entity_id=patient.id,
            patient_priority=patient.priority,  # NEW
        )

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.obstruct_nurse())
        self.env.run(until=self.param.sim_duration)

    def run_warm_up_assessment(self):
        old_warm_up = self.param.warm_up_period
        self.param.warm_up_period = 0
        self.param.sim_duration_warm_up_assessment = (
            self.param.results_collection_period
            * self.param.warm_up_assessment_sim_length_scaler
        )
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.cumulative_mean_tracker())
        self.env.run(until=self.param.sim_duration_warm_up_assessment)
        self.param.warm_up_period = old_warm_up

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dateframe = pd.DataFrame(entity.__dict__ for entity in entity_list)
        return entity_dateframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_nurse = entity_dataframe["q_time_nurse"].mean()
        self.sd_q_time_nurse = entity_dataframe["q_time_nurse"].std()
        self.perc_90_q_time_nurse = entity_dataframe["q_time_nurse"].quantile(0.9)

        self.replication_arrival_times = entity_dataframe["arrival_time"]


class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_nurse = pd.NA
        self.trial_sd_q_time_nurse = pd.NA
        self.trial_perc_90_q_time_nurse = pd.NA
        self.ci_lower_q_time_nurse = pd.NA
        self.ci_upper_q_time_nurse = pd.NA
        self.se_q_time_nurse = pd.NA
        self.warm_up_trial = False
        self.trial_logger = TrialLogger()

    def run_trial(self):
        for replication_id in range(self.param.num_replications):
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

        for wu_replication_id in range(self.param.num_replications_warm_up_assessment):
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

        reference_df = self.list_of_cumulative_mean_dfs[0]
        x_col = "Simulation Time"
        y_cols = [
            col
            for col in reference_df.columns
            if col
            not in [
                x_col,
                "id",
                "arrival_time",
                "priority",  # NEW
            ]
        ]

        for col in y_cols:
            fig = go.Figure()

            for i, df in enumerate(self.list_of_cumulative_mean_dfs, start=1):
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[col],
                        mode="lines",
                        name=f"replication_{i}",
                        line=dict(color="lightblue", width=1),
                    )
                )

            combined = []

            for df in self.list_of_cumulative_mean_dfs:
                combined.append(df[[x_col, col]])

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
                yaxis_title="Cumulative Mean",
            )

            fig.show()
            fig.write_html(f"cumul_mean_{col}.html")

    def calculate_trial_results(self):
        if self.warm_up_trial:
            total_reps = self.param.num_replications_warm_up_assessment
        else:
            total_reps = self.param.num_replications

        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in self.list_of_simulation_replications
        )

        self.trial_mean_q_time_nurse = self.replication_df["mean_q_time_nurse"].mean()

        self.trial_sd_q_time_nurse = self.replication_df["mean_q_time_nurse"].std()

        self.trial_perc_90_q_time_nurse = self.replication_df[
            "mean_q_time_nurse"
        ].quantile(0.9)

        self.se_q_time_nurse = self.trial_sd_q_time_nurse / math.sqrt(total_reps)

        t = stats.t.ppf(0.975, df=total_reps - 1)

        self.ci_lower_q_time_nurse = self.trial_mean_q_time_nurse - (
            t * self.se_q_time_nurse
        )

        self.ci_upper_q_time_nurse = self.trial_mean_q_time_nurse + (
            t * self.se_q_time_nurse
        )

    def plot_arrival_time_frequencies(self):
        self.arrival_times_df = pd.DataFrame(columns=["arrival_time"])

        for replication in self.list_of_simulation_replications:
            self.arrival_times_df = pd.concat(
                [
                    self.arrival_times_df,
                    replication.replication_arrival_times.to_frame(),
                ],
                ignore_index=True,
            )

        self.arrival_times_df["arr_time_bins"] = pd.cut(
            self.arrival_times_df["arrival_time"],
            bins=[i for i in range(0, self.param.sim_duration + 1, 60)],
            include_lowest=True,
            right=False,
        )

        self.arrival_times_df_grouped = (
            self.arrival_times_df.groupby("arr_time_bins").count().reset_index()
        )

        self.arrival_times_df_grouped["arr_times_bins_str"] = (
            self.arrival_times_df_grouped["arr_time_bins"].astype("str")
        )

        self.arrival_times_df_grouped["mean_arrivals_in_period_per_rep"] = (
            self.arrival_times_df_grouped["arrival_time"] / self.param.num_replications
        )

        fig = px.line(
            self.arrival_times_df_grouped,
            x="arr_times_bins_str",
            y="mean_arrivals_in_period_per_rep",
        )

        fig.show()
        fig.write_html("clinic_arrival_time_frequencies.html")


if __name__ == "__main__":
    my_params = Param(
        patient_iat_csv="nspp_example_dataset.csv",
        warm_up_period=0,
        num_replications=3,
        num_nurses=3,
        num_nurses_unav=0,  # Switch off nurse obstruction for this example
        results_collection_period=60 * 24,  # 24 hours
    )
    my_trial = Trial(my_params)
    my_trial.run_trial()
    my_trial.calculate_trial_results()
    print("TRIAL RESULTS")
    print("-----------------------")
    print("Queuing Time for the Nurse")
    print(f"Mean : {my_trial.trial_mean_q_time_nurse:.2f} minutes")
    print(f"SD : {my_trial.trial_sd_q_time_nurse:.2f} minutes")
    print(f"90th Perc : {my_trial.trial_perc_90_q_time_nurse:.2f} minutes")
    print()

    print(my_trial.trial_logger.get_log_by_run(run=1, as_df=True).head(10))

    layout = create_event_position_df(
        [
            EventPosition(event="arrival", x=0, y=350, label="Entrance"),
            EventPosition(
                event="nurse_wait_begins", x=200, y=250, label="Waiting for Nurse"
            ),
            EventPosition(
                event="being_seen_by_nurse",
                x=200,
                y=150,
                label="Being Seen By Nurse",
                resource="num_nurses",
            ),
            EventPosition(event="depart", x=200, y=50, label="Exit"),
        ]
    )

    # NEW
    # A function that will give us a different icon depending on priority
    def show_priority_icon(row):
        # First check this isn't a '+ y more' row
        if "more" not in row["icon"]:
            if row["patient_priority"] == 1:
                return "🚨"
            if row["patient_priority"] == 2:
                return "⚠️"
            else:
                return row["icon"]
        else:
            return row["icon"]

    reshaped_df = reshape_for_animations(
        event_log=my_trial.trial_logger,
        run_number=1,
        every_x_time_units=1,
        limit_duration=my_params.sim_duration,
    )

    animation_df = generate_animation_df(
        full_entity_df=reshaped_df,
        event_position_df=layout,
    )

    # NEW
    # Assign our custom priority icons
    animation_df = animation_df.assign(
        icon=animation_df.apply(show_priority_icon, axis=1)
    )

    fig = generate_animation(
        full_entity_df_plus_pos=animation_df,
        event_position_df=layout,
        scenario=my_params,
    )

    fig.show()

    # Let's see if our TrialLogger can give us queue size insight
    queue_fig = my_trial.trial_logger.plot_queue_size(
        ["nurse_wait_begins"],
        limit_duration=my_params.sim_duration,
        # This doesn't have to be at the same granularity as our animation
        every_x_time_units=5,
    )
    queue_fig.show()

    for priority in [1, 2, 3]:
        filtered_event_log = add_sim_timestamp(
            my_trial.trial_logger.get_log_by_run(run=1, as_df=True),
            time_unit="minutes",
            sim_start="09:00:00",
        )

        if priority != "all":
            filtered_event_log = filtered_event_log[
                filtered_event_log["patient_priority"] == priority
            ]

        # Now we'll discover the pathways in the model
        nodes, edges = discover_dfg(
            filtered_event_log,
            # Our 'case_col' will be 'entity_id' if we've used EventLogger
            # This just means that each person is considered to be a separate
            # journey
            case_col="entity_id",
        )

        # A static representation of flow through the process
        graphviz_graph = dfg_to_graphviz(
            nodes,
            edges,
            min_frequency=5,
            title=f"Priority: {priority}",  # NEW
        )
        display(graphviz_graph)
