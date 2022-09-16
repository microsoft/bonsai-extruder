import json
import os
import random
import matplotlib.pyplot as plt
import ast
import requests
import numpy as np
import time as tm
import pandas as pd

from bonsai_common import SimulatorSession, Schema
#import dotenv
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from sim import extrusion_model as em
from sim import units


# time step (seconds) between state updates
Δt = 1


class ExtruderSimulation():
    def reset(
        self,
        noise: float = 0.0001,
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
    ):
        """
        Extruder model for simulation.

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

        self.noise = noise
        #self.noise = 0.01
        
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
        

    def episode_start(self, config: Schema) -> None:
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

    def step(self):

        # add a small amount of random noise to the actions to avoid
        # the trivial solution of simply applying zero acceleration
        # on each iteration
        σ_max = self.noise 
        σ_s = random.uniform(-σ_max, σ_max)
        σ_c = random.uniform(-σ_max, σ_max)

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

    def episode_step(self) -> None:
        #conect to Bonsai Brain - Connect to Monolithic Brain 
        payload = self.get_state()
        url = "http://localhost:5000/v1/prediction"
        try:
            r = ast.literal_eval(requests.post(url, data=json.dumps(payload)).text)
            Δws = r["screw_angular_acceleration"]
            Δfc = r["cutter_acceleration"]
            
        except:
            Δws = 0
            Δfc = 0
            print("Error to get brain value")
      
        self.Δω_s = Δws
        self.Δf_c = Δfc
        self.step()

    def get_state(self):
        return {
            "screw_angular_speed": self.ω_s,
            "screw_angular_acceleration": self.Δω_eff,
            "cutter_frequency": self.f_c,
            "cutter_acceleration": self.Δf_eff,
            "temperature": self.T,
            "product_length": self.L,
            "flow_rate": self.Q,
            "yield": self.yield_,
            "efficiency": self.efficiency,
            "demand": self.demand,
            "total_cost": float(self.total_cost),
            "total_revenue": float(self.total_revenue),
            "total_profit": float(self.total_profit),
            "parts_on_spec": float(self.parts_on_spec),
            "cnt": float(self.cnt)
        }

    def halted(self) -> bool:
        if self.L < 0:
            return True
        else:
            return False

    def get_interface(self) -> SimulatorInterface:
        """Register sim interface."""

        with open("interface.json", "r") as infile:
            interface = json.load(infile)

        return SimulatorInterface(
            name=interface["name"],
            timeout=interface["timeout"],
            simulator_context=self.get_simulator_context(),
            description=interface["description"],
        )


def main():
    file_name = 'state_space_data_controller_bonsai.csv'
    try:
        df = pd.read_csv(file_name)
    except:
        df = pd.DataFrame()
        
    onspec = 0
    # starting time
    start = tm.time()

    extruder_sim = ExtruderSimulation()

    ws0_list = [3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.5, 7.5, 7.5]
    fc0_list = [0.3,0.2,0.12,0.375,0.3,0.2, 0.1, 0.2, 0.325]
    demand_list = [75,75,75,120,120,120,190,190,190]
    i = 0 #demand scenario index
    
    extruder_sim.reset(noise = 0.0001,
        ω0_s= ws0_list[i],
        Δω0_s = 0,
        f0_c= fc0_list[i],
        Δf0_c= 0,
        T= units.celsius_to_kelvin(190),
        demand = demand_list[i], 
        cost = 0.86, 
        selling_price = 2.0,
        L0= 1 * 12 * units.METERS_PER_INCH,
        ε= 0.1 * units.METERS_PER_INCH
                       )

    ws_l = []
    fc_l = []
    T_l = []
    L_l = []
    parts_on_spec_l = []
    eff_list = []
    profit_l = []
    model_name = f'bonsai_{ws0_list[i]}_{fc0_list[i]}_{demand_list[i]}'

    time = 600 
    graphics = True
    if graphics:
        plt.figure(figsize=(10,7))
        plt.ion()
    
    for k in range(time):
        #break the simulation for Thermal runaway
        if extruder_sim.halted():
            break
        
        extruder_sim.episode_step()
        state = extruder_sim.get_state()

        L1 = 0.3048 - 0.00254
        L2 = 0.3048 + 0.00254
        if extruder_sim.L >= L1 or extruder_sim.L <= L2:
            onspec = 1
        else:
            onspec = 0
        #generate dataframe
        df = df.append(pd.DataFrame({'ws':[state["screw_angular_speed"]],'fc':[state["cutter_frequency"]],'L':[state["product_length"]],
                                     'Q':[0], 'yield':[extruder_sim.yield_], 'T':[state['temperature']],'demand':[extruder_sim.demand],
                                     'parts_on_spec':[extruder_sim.parts_on_spec],
                                     'efficiency':[extruder_sim.efficiency],'profit':[extruder_sim.total_profit], 'time': [k],
                                     'total_revenue': [extruder_sim.total_revenue], 'model_name':[model_name]
                                     }))
        
        ws_l.append(state["screw_angular_speed"])
        fc_l.append(state["cutter_frequency"])
        T_l.append(state['temperature'])
        L_l.append(state["product_length"])
        parts_on_spec_l.append(extruder_sim.parts_on_spec)
        eff_list.append(extruder_sim.efficiency)
        profit_l.append(state["total_profit"])

        if graphics:
            plt.clf()
            
            plt.subplot(6,1,1)
            plt.plot([i for i in range(len(ws_l))],ws_l,'k.-',lw=2)
            plt.ylabel('Screw Angular Speed')
            plt.legend(['Screw Angular Speed'],loc='best')
            

            plt.subplot(7,1,2)
            plt.plot([i for i in range(len(fc_l))],fc_l,'k.-',lw=2)
            plt.ylabel('Cutter Frequency')
            plt.legend(['Cutter Frequency'],loc='best')
            
            plt.subplot(7,1,3)
            plt.plot([i for i in range(len(L_l))],L_l,'k.-',lw=2)
            plt.ylabel('Length')
            plt.legend(['Length'],loc='best')

            plt.subplot(7,1,4)
            plt.plot([i for i in range(len(T_l))],T_l,'k.-',lw=2)
            plt.ylabel('Temperature')
            plt.legend(['Temperature'],loc='best')

            plt.subplot(7,1,5)
            plt.plot([i for i in range(len(parts_on_spec_l))],parts_on_spec_l,'k.-',lw=2)
            plt.plot([i for i in range(len(parts_on_spec_l))],[state['demand'] for i in range(len(parts_on_spec_l))],'r-',lw=2)
            plt.ylabel('Parts on Spec')
            plt.legend(['Parts on Spec','Demand'],loc='best')

            plt.subplot(7,1,6)
            plt.plot([i for i in range(len(eff_list))],eff_list,'b.-',lw=2)
            plt.ylabel('Efficiency')
            plt.legend(['Efficiency'],loc='best')

            plt.subplot(7,1,7)
            plt.plot([i for i in range(len(profit_l))],profit_l,'g.-',lw=2)
            plt.plot([i for i in range(len(profit_l))],[0 for i in range(len(profit_l))],'k-',lw=1)
            plt.ylabel('Profit $')
            plt.legend(['Profit'],loc='best')

            
            plt.xlabel('Time (second)')
            #plt.legend(['Temperature Setpoint','Reactor Temperature'],loc='best')
            

            plt.draw()
            plt.pause(0.001)

    print("Parts produced: ", parts_on_spec_l[-1])
    print("Maximum Temperature: ", max(T_l))
    print("Maximum Frequency: ", max(fc_l))
    print("Total Parts: ", extruder_sim.total_parts)
    print("Efficiency: ", extruder_sim.efficiency)
    print("Profit: ", extruder_sim.total_profit)
    # end time
    end = tm.time()
    time_run = end - start
    # total time taken
    print(f"Runtime of the program is {time_run} seconds")
    
    df.to_csv(file_name,index=False)


if __name__ == "__main__":
    main()
