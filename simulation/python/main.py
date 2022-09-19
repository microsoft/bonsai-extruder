import json
import os
import random
import ast
from scipy import interpolate
import math
from typing import Any, Dict, Union
import time
import pathlib
from functools import partial
import datetime

import dotenv

from azure.core.exceptions import HttpResponseError

from microsoft_bonsai_api.simulator.client import (BonsaiClient,
                                                   BonsaiClientConfig)
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface, SimulatorState)


from sim import extrusion_model as em
from sim import units


import numpy as np
import sys

import matplotlib.pyplot as plt

import requests

# time step (seconds) between state updates
Δt = 1

LOG_PATH = "logs"

def ensure_log_dir(log_full_path):
    """
    Ensure the directory for logs exists — create if needed.
    """
    print(f"logfile: {log_full_path}")
    logs_directory = pathlib.Path(log_full_path).parent.absolute()
    print(f"Checking {logs_directory}")
    if not pathlib.Path(logs_directory).exists():
        print(
            "Directory does not exist at {0}, creating now...".format(
                str(logs_directory)
            )
        )
        logs_directory.mkdir(parents=True, exist_ok=True)
        
class Simulation:
    def __init__(self,
                modeldir: str = "sim",
                render: bool = False,
                log_data: bool = False,
                log_file_name: str = None,
                env_name: str = "PVC_Extruder",
                 ):
        self.reset()
        self.terminal = False
        self.render = render
        self.log_data = log_data
        if not log_file_name:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_name = current_time + "_" + env_name + "_log.csv"

        self.log_full_path = os.path.join(LOG_PATH, log_file_name)
        ensure_log_dir(self.log_full_path)
        
    def reset(
        self,
        noise: float = 0,
        ω0_s: float = 1e-6,
        Δω0_s: float = 0,
        f0_c: float = 1e-6,
        Δf0_c: float = 0,
        T: float = units.celsius_to_kelvin(190),
        demand: float = 0.1,
        cost: float = 0,
        selling_price: float = 0,
        L0: float = 1 * 12 * units.METERS_PER_INCH,
        ε: float = 0.1 * units.METERS_PER_INCH,
        config: dict = None
    ):
        """Helper function for resetting a simulator environment
        Parameters
        ----------
        config : dict, optional
            [description], by default None
        Extruder Model for simulation

        Parameters
        ----------
        ω0_s : float, optional
            Initial screw angular speed (radians / second).
        Δω0_s : float, optional
            Initial change in screw angular speed (radians / second^2).
        f0_c : float, optional
            Initial cutter frequency (hertz).
        Δf0_c : float, optional
            Initial change in cutter frequency (1 / second^2).
        T : float, optional
            Initial temperature (Kelvin).
        L0 : float, optional
            Initial product length (meters).
        ε : float, optional
            Product tolerance (meters).
        """
        
        self.default_config = {
            "noise": 0,
            "initial_screw_angular_speed": 1e-6,
            "initial_screw_angular_acceleration": 0,
            "initial_cutter_frequency": 1e-6,
            "initial_cutter_acceleration": 0,
            "initial_temperature": 463,
            "demand": 0.1,
            "cost": 0,
            "selling_price": 0
            }
        
        if config:
            self.sim_config = config
        else:
            self.sim_config = self.default_config
            
        self.noise = noise
        
        # angular speed of the extruder screw (radians / second)
        self.ω_s = ω0_s

        # change in angular speed of the extruder screw (radians / second^2)
        self.Δω_s = Δω0_s
        self.Δω_eff = self.Δω_s

        # frequency of the cutter (hertz)
        self.f_c = f0_c

        # change in cutter frequency (1 / second^2)
        self.Δf_c = Δf0_c
        self.Δf_eff = self.Δf_c

        self.demand = demand

        # temperature (Kelvin)
        self.T = T

        self.L0 = L0
        self.ε = ε

        model = em.ExtrusionModel(
            ω=self.ω_s, Δω=self.Δω_s, f_c=self.f_c, T=self.T, Δt=Δt
        )

        self.T += model.ΔT

        # material flow rate (meters^3 / second)
        self.Q = model.Q_op

        # product length (meters)
        self.L = model.L

        # manufacturing yield, defined as the number of good parts
        # per iteration (dimensionless)
        self.yield_ = model.yield_

        self.parts_per_iteration = 0
        self.total_parts = 0
        self.parts_on_spec = 0
        # efficiency
        self.efficiency = 0

        self.cost = cost
        self.selling_price = selling_price
        self.total_cost = 0
        self.total_revenue = 0
        self.total_profit = 0
        self.cnt = 0
  

    def episode_start(self, config: dict = None):
        """Method invoked at the start of each episode with a given 
        episode configuration.
        Parameters
        ----------
        config : Dict[str, Any]
            SimConfig parameters for the current episode defined in Inkling
        """
        self.config = config
            
        self.reset(
            noise = config.get("noise"),
            ω0_s=config.get("initial_screw_angular_speed"),
            Δω0_s=config.get("initial_screw_angular_acceleration"),
            f0_c=config.get("initial_cutter_frequency"),
            Δf0_c=config.get("initial_cutter_acceleration"),
            T=config.get("initial_temperature"),
            demand = config.get("demand"),
            cost = config.get("cost"),
            selling_price = config.get("selling_price")
            )

    def step(self, Δω_s: float, Δf_c: float):
        # add a small amount of random noise to the actions to avoid
        # the trivial solution of simply applying zero acceleration
        # on each iteration
        σ_max = self.noise #0.0001
        σ_s = random.uniform(-σ_max, σ_max)
        σ_c = random.uniform(-σ_max, σ_max)

        self.Δω_s = Δω_s
        self.Δf_c = Δf_c

        self.Δω_eff = self.Δω_s * (1 + σ_s)
        self.ω_s += Δt * self.Δω_eff

        self.Δf_eff = self.Δf_c * (1 + σ_c)
        self.f_c += Δt * self.Δf_eff

        model = em.ExtrusionModel(
            ω=self.ω_s, Δω=self.Δω_eff, f_c=self.f_c, T=self.T, Δt=Δt
        )

        self.T += model.ΔT

        # material flow rate (meters^3 / second)
        self.Q = model.Q_op

        # material density (kg / m^3)
        self.pho = 1.38 * 1000

        # mass flow rate in kg/s
        self.Q_m = self.Q * self.pho

        # degradation
        if self.T > 478:
            self.Q_m = self.Q_m * 0.9565
        elif self.T > 500:
            self.Q_m = self.Q_m * (-0.0036*self.T + 2.7565)

        # product length (meters)
        self.L = model.L

        # manufacturing yield, defined as the number of good parts
        # per iteration (dimensionless)
        self.yield_ = model.yield_

        self.parts_per_iteration = self.f_c * Δt

        self.total_parts += self.parts_per_iteration

        # total cost ($)
        self.total_cost += self.cost * self.Q_m

        if abs(self.L - self.L0) < self.ε:
            self.parts_on_spec += self.yield_
            if self.parts_on_spec <= self.demand :
                # total revenue ($) assuming that we are selling all the good parts
                self.total_revenue += self.selling_price * self.Q_m
            
        # total profit ($) assuming that we are selling all the good parts
        self.total_profit = (self.total_revenue - self.total_cost)

        self.efficiency = float(self.parts_on_spec/self.total_parts)

        self.cnt += 1

    def episode_step(self, action: Dict[str, Any]):
        """Called for each step of the episode 
        Parameters
        ----------
        action : Dict[str, Any]
            BrainAction chosen from the Bonsai Service, prediction or exploration
        """
        
        self.step(action.get("screw_angular_acceleration"), action.get("cutter_acceleration"))
        

    def get_state(self):
        """ Called to retreive the current state of the simulator. """
        sim_state = {
            "screw_angular_speed": float(self.ω_s),
            "screw_angular_acceleration": float(self.Δω_eff),
            "cutter_frequency": float(self.f_c),
            "cutter_acceleration": float(self.Δf_eff),
            "temperature": float(self.T),
            "product_length": float(self.L),
            "flow_rate": float(self.Q),
            "yield": float(self.yield_),
            "efficiency": float(self.efficiency),
            "demand": float(self.demand),
            "total_cost": float(self.total_cost),
            "total_revenue": float(self.total_revenue),
            "total_profit": float(self.total_profit),
            "parts_on_spec": float(self.parts_on_spec),
            "cnt": float(self.cnt)
        }
        return sim_state
    
    def halted(self) -> bool:
        if self.L < 0:
            return True
        else:
            return False

    def get_interface(self, config_client):
        """Register sim interface."""

        with open("interface.json", "r") as infile:
            interface = json.load(infile)

        return SimulatorInterface(
            name=interface["name"],
            timeout=interface["timeout"],
            simulator_context=config_client.simulator_context,
            description=interface["description"],
        )

    def log_iterations(self, state, action, episode: int = 1, iteration: int = 1):
        """Log iterations during training to a CSV.
        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """

        import pandas as pd

        # def add_prefixes(d, prefix: str):
        #     return {f"{prefix}_{k}": v for k, v in d.items()}
        # state = add_prefixes(state, "state")
        # action = add_prefixes(action, "action")
        # config = add_prefixes(self.sim_config, "config")
        config = self.sim_config
        data = {**state, **action, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if episode == 1 and iteration == 1:
            # Store initial states and configs because we don't have actions yet
            print('Collecting episode start logdf, waiting for action keys from episode step')
            self.initial_log = data
        elif iteration >= 2:
            if os.path.exists(self.log_full_path):
                # Check if we've alrdy written to a file with 2 rows, continue writing
                log_df = pd.DataFrame({k: log_df[k] for k in self.desired_dict_order})
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="a", header=False, index=False
                )
            else:
                # Take intial states and configs from ep 1, update with actions with None
                for key, val in action.items():
                    self.initial_log[key] = None
                self.initial_log = pd.DataFrame(self.initial_log, index=[0])
                self.action_keys = action.keys()

                log_df = pd.concat([self.initial_log, log_df], sort=False)
                log_df.to_csv(
                    path_or_buf=self.log_full_path, mode="w", header=True, index=False
                )
                if episode == 1 and iteration == 2:
                    self.desired_dict_order = log_df.keys()
        elif iteration == 1:
            # Now every episode start will use action keys and reorder dict properly
            for key in self.action_keys:
                log_df[key] = None
            log_df = pd.DataFrame({k: log_df[k] for k in self.desired_dict_order})
            log_df.to_csv(
                path_or_buf=self.log_full_path, mode="a", header=False, index=False
            )
        else:
            print('Something else went wrong with logs')
            exit()

def main():

    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    # values in `.env`, if they exist, take priority over environment variables
    dotenv.load_dotenv(".env", override=True)

    if workspace is None:
        raise ValueError("The Bonsai workspace ID is not set.")
    if access_key is None:
        raise ValueError("The access key for the Bonsai workspace is not set.")

    config = BonsaiClientConfig(workspace=workspace, access_key=access_key)
    cstr_sim = Simulation(config)
    cstr_sim.reset()
    while cstr_sim.run():
        continue

def main(
    render: bool=False,
    log_iterations: bool=False,
    config_setup: bool=False,
    env_file: Union[str, bool]=".env",
    workspace: str=None,
    accesskey: str=None,
):
    """Main entrypoint for running simulator connections
    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    config_setup: bool, optional
        if enabled then uses a local `.env` file to find sim workspace id and access_key
    env_file: str, optional
        if config_setup True, then where the environment variable for lookup exists
    workspace: str, optional
        optional flag from CLI for workspace to override
    accesskey: str, optional
        optional flag from CLI for accesskey to override
    """

    # check if workspace or access-key passed in CLI
    use_cli_args = all([workspace, accesskey])

    # use dotenv file if provided
    use_dotenv = env_file or config_setup

    # check for accesskey and workspace id in system variables
    # Three scenarios
    # 1. workspace and accesskey provided by CLI args
    # 2. dotenv provided
    # 3. system variables
    # do 1 if provided, use 2 if provided; ow use 3; if no sys vars or dotenv, fail

    if use_cli_args:
        # BonsaiClientConfig will retrieve as environment variables
        os.environ["SIM_WORKSPACE"] = workspace
        os.environ["SIM_ACCESS_KEY"] = accesskey
    elif use_dotenv:
        if not env_file:
            env_file = ".env"
        print(
            f"No system variables for workspace-id or access-key found, checking in env-file at {env_file}"
        )
        workspace, accesskey = env_setup(env_file)
        load_dotenv(env_file, verbose=True, override=True)
    else:
        try:
            workspace = os.environ["SIM_WORKSPACE"]
            accesskey = os.environ["SIM_ACCESS_KEY"]
        except:
            raise IndexError(
                f"Workspace or access key not set or found. Use --config-setup for help setting up."
            )

    # Grab standardized way to interact with sim API
    #sim = TemplateSimulatorSession(render=render, log_data=log_iterations)
    

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Create simulator session and init sequence id
    sim = Simulation(render=render, log_data=log_iterations)
    registration_info = sim.get_interface(config_client)
    '''registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=60,
        simulator_context=config_client.simulator_context,
    )'''
    

    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex
                )
            )
            raise ex
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    registered_session, sequence_id = CreateSession(registration_info, config_client)
    episode = 0
    iteration = 1

    try:
        while True:
            # Advance by the new state depending on the event type
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action={},
                    )
            elif event.type == "EpisodeStep":
                sim.episode_step(event.episode_step.action)
                iteration += 1
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action=event.episode_step.action,
                    )
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 1
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":
    main(
            config_setup=False,
            render=False,
            log_iterations=False,
            env_file=None,
            workspace=None,
            accesskey=None,
        )
