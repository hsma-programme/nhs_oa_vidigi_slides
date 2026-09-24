"""
This is the one-step model from the second simpy session (2B)
"""

import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
from vidigi.logging import EventLogger  # NEW
from vidigi.utils import create_event_position_df, EventPosition  # NEW


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
        self.nurse = simpy.Resource(self.env, capacity=self.param.num_nurses)
        self.patient_inter_dist = Exponential(mean=self.param.mean_patient_inter)
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time,
        )
        self.list_of_patients = []
        self.mean_q_time_nurse = pd.NA
        self.sd_q_time_nurse = pd.NA
        self.perc_90_q_time_nurse = pd.NA

        self.logger = EventLogger(env=self.env)  # NEW

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.list_of_patients.append(p)
            self.env.process(self.attend_clinic(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def attend_clinic(self, patient):
        self.logger.log_arrival(entity_id=patient.id)  # NEW

        start_q_nurse = self.env.now

        self.logger.log_queue(entity_id=patient.id, event="nurse_wait_begins")  # NEW

        with self.nurse.request() as req:
            yield req
            end_q_nurse = self.env.now
            self.logger.log_queue(entity_id=patient.id, event="being_seen_by_nurse")  # NEW
            patient.q_time_nurse = end_q_nurse - start_q_nurse

            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)
            self.logger.log_queue(entity_id=patient.id, event="nurse_treatment_ends")  # NEW

        self.logger.log_departure(entity_id=patient.id)  # NEW

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


# NEW
# The special line
# if __name__ == "main"
# tells Python to only run the bit below if you're running the full script
# in the terminal or interactive window
# This just makes our file more robust if in future we wanted to reuse our classes
# elsewhere, and it's good practice to do so
if __name__ == "__main__":
    my_params = Param()
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
            ),
            # We don't need to visualise the 'nurse_treatment_ends' step as the timing will
            # be identical to the depart step
            EventPosition(event="depart", x=200, y=50, label="Exit"),
        ]
    )

    print(my_model.logger.to_dataframe().head(10))

    fig = my_model.logger.animate_activity_log(
        event_position_df=layout,
        every_x_time_units=1,
    )

    fig.show()

    # Optionally, we could output these to files
    # my_model.logger.to_csv("simplest_event_log.csv", index=False)
    # fig.write_html("simplest_animation.html")
    # END NEW #
