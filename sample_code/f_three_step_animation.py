"""
This is the one-step model from the second simpy session (2B)
"""

import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
from vidigi.logging import EventLogger, TrialLogger
from vidigi.utils import create_event_position_df, EventPosition
from vidigi.prep import reshape_for_animations, generate_animation_df  # NEW
from vidigi.animation import generate_animation  # NEW
from vidigi.resources import VidigiStore


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
        self.logger = EventLogger(
            env=self.env,
            run_number=self.replication_id,
        )
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
    # Rather than calling animate_activity_log() in one go, we now run
    # the three steps that it is made up of, one at a time

    # Step 1: build the minute-by-minute snapshots of where everyone is
    reshaped_df = reshape_for_animations(
        event_log=my_trial.trial_logger,
        run_number=1,
        every_x_time_units=1,
        limit_duration=my_params.sim_duration,
    )

    # Step 2: assign each entity an icon and a position for every snapshot
    animation_df = generate_animation_df(
        full_entity_df=reshaped_df,
        event_position_df=layout,
    )

    # Step 3: turn it into an animation
    fig = generate_animation(
        full_entity_df_plus_pos=animation_df,
        event_position_df=layout,
        scenario=my_params,
    )

    fig.show()
