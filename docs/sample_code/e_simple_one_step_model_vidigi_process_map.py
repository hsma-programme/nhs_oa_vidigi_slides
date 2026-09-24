"""
This is the one-step model from the second simpy session (2B)
"""

import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
from vidigi.logging import EventLogger, TrialLogger
from vidigi.utils import create_event_position_df, EventPosition
from vidigi.resources import VidigiStore

# NEW imports
from vidigi.process_mapping import (
    add_sim_timestamp,
    discover_dfg,
    dfg_to_graphviz,
    dfg_to_cytoscape,
)
from IPython.display import display


class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = pd.NA


class Param:
    def __init__(
        self,
        mean_patient_inter=5,
        mean_nurse_consult_time=6,
        sd_nurse_consult_time=1,
        num_nurses=1,
        sim_duration=120,
        num_replications=5,
    ):
        self.mean_patient_inter = mean_patient_inter
        self.mean_nurse_consult_time = mean_nurse_consult_time
        self.sd_nurse_consult_time = sd_nurse_consult_time
        self.num_nurses = num_nurses
        self.sim_duration = sim_duration
        self.num_replications = num_replications


class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.logger = EventLogger(env=self.env, run_number=self.replication_id)
        self.nurse = VidigiStore(
            self.env,
            num_resources=self.param.num_nurses,
            label="nurse",
            logger=self.logger,
        )
        self.patient_inter_dist = Exponential(mean=self.param.mean_patient_inter)
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time,
        )
        self.list_of_patients = []
        self.mean_q_time_nurse = pd.NA
        self.sd_q_time_nurse = pd.NA
        self.perc_90_q_time_nurse = pd.NA

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.list_of_patients.append(p)
            self.env.process(self.attend_clinic(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def attend_clinic(self, patient):
        self.logger.log_arrival(entity_id=patient.id)

        start_q_nurse = self.env.now

        self.logger.log_queue(entity_id=patient.id, event="nurse_wait_begins")

        with self.nurse.request(
            entity_id=patient.id,
            start_event="being_seen_by_nurse",
            end_event="nurse_treatment_ends",
        ) as req:
            yield req
            end_q_nurse = self.env.now
            patient.q_time_nurse = end_q_nurse - start_q_nurse
            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)

        self.logger.log_departure(entity_id=patient.id)

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dateframe = pd.DataFrame(entity.__dict__ for entity in entity_list)
        return entity_dateframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_nurse = entity_dataframe["q_time_nurse"].mean()
        self.sd_q_time_nurse = entity_dataframe["q_time_nurse"].std()
        self.perc_90_q_time_nurse = entity_dataframe["q_time_nurse"].quantile(0.9)


class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_nurse = pd.NA
        self.trial_sd_q_time_nurse = pd.NA
        self.trial_perc_90_q_time_nurse = pd.NA
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
            self.trial_logger.add_log(model_replication.logger)

    def calculate_trial_results(self):
        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in self.list_of_simulation_replications
        )

        self.trial_mean_q_time_nurse = self.replication_df["mean_q_time_nurse"].mean()

        self.trial_sd_q_time_nurse = self.replication_df["mean_q_time_nurse"].std()

        self.trial_perc_90_q_time_nurse = self.replication_df[
            "mean_q_time_nurse"
        ].quantile(0.9)


if __name__ == "__main__":
    my_params = Param(mean_patient_inter=3, num_nurses=2, mean_nurse_consult_time=10)
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

    # We've commented out the animation code as we don't need it for now
    # layout = create_event_position_df(
    #     [
    #         EventPosition(event="arrival", x=0, y=350, label="Entrance"),
    #         EventPosition(
    #             event="nurse_wait_begins", x=200, y=250, label="Waiting for Nurse"
    #         ),
    #         EventPosition(
    #             event="being_seen_by_nurse",
    #             x=200,
    #             y=150,
    #             label="Being Seen By Nurse",
    #             resource="num_nurses",
    #         ),
    #         EventPosition(event="depart", x=200, y=50, label="Exit"),
    #     ]
    # )

    # fig = my_trial.trial_logger.animate_activity_log(
    #     run_number=1,
    #     event_position_df=layout,
    #     every_x_time_units=1,
    #     scenario=my_params,
    # )
    # fig.show()

    # NEW
    # Example 1 - step by step

    my_event_log_timestamp = add_sim_timestamp(
        my_trial.trial_logger.get_log_by_run(run=1, as_df=True),
        time_unit="minutes",
        sim_start="09:00:00",
    )

    # If we print this, we can see our new timestamp column
    print(my_event_log_timestamp.head(10))

    # Now we'll discover the pathways in the model
    nodes, edges = discover_dfg(
        my_event_log_timestamp,
        # Our 'case_col' will be 'entity_id' if we've used EventLogger
        # This just means that each person is considered to be a separate
        # journey
        case_col="entity_id",
    )

    # NOTE: display() only draws these graphs in an interactive environment
    # (e.g. a Jupyter notebook or VSCode's interactive window). If you run this
    # as a plain script from the terminal, you'll just see text.

    # A static representation of flow through the process
    graphviz_graph = dfg_to_graphviz(nodes, edges, min_frequency=5)
    display(graphviz_graph)

    # An interactive version
    cytoscape_widget = dfg_to_cytoscape(
        nodes,
        edges,
        min_frequency=5,
        layout_name="dagre",
        layout_orientation="LR",
        spacing_factor=2,
        width=1400,
    )
    display(cytoscape_widget)

    # NEW
    # Repeat the same thing with the simplified call from the trial logger object

    display(
        my_trial.trial_logger.generate_dfg(
            run_number=1, input_time_format="minutes", output_format="graphviz-object"
        )
    )

    display(
        my_trial.trial_logger.generate_dfg(
            run_number=1, input_time_format="minutes", output_format="cytoscape-jupyter"
        )
    )
