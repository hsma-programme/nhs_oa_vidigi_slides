"""
This is the one-step model from the second simpy session (2B)
"""

import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
from vidigi.logging import EventLogger
from vidigi.utils import create_event_position_df, EventPosition
from vidigi.animation import animate_activity_log
from vidigi.resources import VidigiStore  # NEW


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
    def __init__(self, param):
        self.param = param
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.logger = EventLogger(env=self.env)  # UPDATED - moved above our resource

        # NEW
        # We change simpy.Resource to VidigiStore
        # and we change 'capacity' to 'num_resources' (as capacity has its own
        # special meaning in vidigi's resources)
        # We also pass in a label and our logger (so we need to define our resources
        # after we set up the logger first)

        self.nurse = VidigiStore(
            self.env,
            num_resources=self.param.num_nurses,
            logger=self.logger,
            label="nurse",
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

        # NEW/UPDATED
        # We need to pass in our patient ID and names to use for the events that will be associated
        # with starting to use a resource and finishing using it
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


if __name__ == "__main__":
    # NEW
    # We'll override the number of nurses so we can more clearly see what's going on
    my_params = Param(num_nurses=2)

    my_model = Model(my_params)
    my_model.run_model()

    patient_df = my_model.convert_entity_list_to_dataframe(my_model.list_of_patients)
    my_model.calculate_run_results(patient_df)

    print(
        f"Mean queuing time for the nurse was {my_model.mean_q_time_nurse:.2f}",
        "minutes",
    )
    print(
        f"SD queuing time for the nurse was {my_model.sd_q_time_nurse:.2f}", "minutes"
    )
    print(
        "90th percentile queuing time for the nurse was",
        f"{my_model.perc_90_q_time_nurse:.2f} minutes",
    )

    print(my_model.logger.to_dataframe().head(10))

    # UPDATED - our layout now says which resource to draw
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
                # NEW
                # We now just need to pass in the resource to visualise
                # This will be looked up from our Params class, so we need
                # to make sure the name exactly matches how it's written there
                resource="num_nurses",  # NEW
            ),
            # We **still** don't need to visualise the 'nurse_treatment_ends' step as
            # the timing will be identical to the depart step
            EventPosition(event="depart", x=200, y=50, label="Exit"),
        ]
    )

    fig = my_model.logger.animate_activity_log(
        event_position_df=layout,
        every_x_time_units=1,
        scenario=my_params,  # NEW
        custom_resource_icon="👩‍⚕️",  # NEW - OPTIONAL
        resource_icon_size=32,  # NEW - OPTIONAL
    )

    fig.show()
