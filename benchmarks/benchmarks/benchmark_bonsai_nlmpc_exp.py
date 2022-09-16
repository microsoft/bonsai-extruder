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

from scipy.integrate import odeint
from gekko import GEKKO
from scipy import interpolate
import math


# time step (seconds) between state updates
Δt = 1

class linear_mpc:
    def __init__(self, ω0_s,f0_c,Δω0_s,Δf0_c, noise):
        w_s0 = ω0_s
        f_c0 = f0_c
        self.noise = noise
        
        L0 = 0.3
        T0 = 190+273
        L_ss = 0.3048
        T_ss = T0
        u_ss = 0
        self.x0 = np.empty(2)
        self.x0[0] = L_ss
        self.x0[1] = T_ss
        self.T_ = T0

        # GEKKO linear MPC
        m = GEKKO(remote=False) #True for Mac and Linux, False for Windows

        m.time = np.linspace(0, 200, num=200) #seconds

        # initial conditions
        tau1 = m.Const(value = 1)
        tau2 = m.Const(value = 1)

        # Manipulated and Controlled Variables
        m.w_s = m.MV(value = w_s0) 
        m.f_c = m.MV(value = f_c0) #mv with constraints

        self.extruder_sim_mpc = em.ExtrusionModel(
            ω=w_s0, Δω=Δω0_s, f_c=f_c0, T=self.T_, Δt=1
        )

        m.L = m.CV(value = self.extruder_sim_mpc.L) #cv

        # Process dynamic model
        c = [ 2.36975901, 16.80446811, 0.04710705]
        self.c = c
        m.Equation([m.L + m.L.dt() == c[0]*m.exp(-c[1]*m.f_c) + c[2]* m.w_s]) #NLMPC Exp


        #MV tuning - Cooling Temp
        m.w_s.STATUS = 1
        m.w_s.FSTATUS = 0
        m.w_s.UPPER = 8.3
        m.w_s.LOWER = 2.1
        m.w_s.DMAXHI = 0.02094395  # constrain movement
        m.w_s.DMAXLO = -0.02094395  

        m.f_c.STATUS = 1
        m.f_c.FSTATUS = 0
        m.f_c.UPPER = 0.4
        m.f_c.LOWER = 0.06
        m.f_c.DMAXHI = 0.01  # constrain movement
        m.f_c.DMAXLO = -0.01 

        m.L.STATUS = 1
        m.L.FSTATUS = 1
        m.L.SP = 0.3048
        m.L.TR_INIT = 2
        m.L.TAU = 1.0 

        m.options.CV_TYPE = 2 # the objective is an l2-norm (squared error) 
        m.options.IMODE = 6 # MPC
        m.options.SOLVER = 3

        # time Interval (min)
        time = 600+1 
        t = np.linspace(0,time, time)

        #Store results for plotting
        self.L = np.ones(len(t)) * L_ss
        self.T = np.ones(len(t)) * T_ss
        self.u1 = np.ones(len(t)) * u_ss
        self.u2 = np.ones(len(t)) * u_ss

        self.u1[0] = w_s0
        self.u2[0] = f_c0

        self.m = m
        
        self.Q = self.extruder_sim_mpc.Q_op  

        self.T_ += self.extruder_sim_mpc.ΔT
        
    def step(self,cnt, T, L):
        
        if cnt == 0:
            du1 = 0
            du2 = 0
        else:          
            du1 = self.u1[cnt] - self.u1[cnt-1]
            du2 = self.u2[cnt] - self.u2[cnt-1]

        i = cnt

        # add a small amount of random noise to the actions to avoid
        # the trivial solution of simply applying zero acceleration
        # on each iteration
        σ_max = self.noise 
        σ_s = random.uniform(-σ_max, σ_max)
        σ_c = random.uniform(-σ_max, σ_max)

        self.u1[cnt] = self.u1[cnt] * (1 + σ_s)
        self.u2[cnt] = self.u2[cnt] * (1 + σ_c)
        
        self.extruder_sim_mpc = em.ExtrusionModel(
            ω=self.u1[cnt], Δω=du1, f_c=self.u2[cnt], T=self.T_, Δt=1
        )

        self.T_ += self.extruder_sim_mpc.ΔT
        self.Q = self.extruder_sim_mpc.Q_op

        # retrieve measurements
        self.T[i+1] = self.T_
        self.L[i+1] = self.extruder_sim_mpc.L
        # insert measurement
        self.m.L.MEAS = self.L[i+1]
        # update setpoint
        try:
            self.m.solve(disp=False)
        except:
            print('Solution not found')

        # change to a fixed starting point for trajectory
        self.m.L.TR_INIT = 2
        # retrieve new values
        self.u1[i+1] = self.m.w_s.NEWVAL
        self.u2[i+1] = self.m.f_c.NEWVAL
        # update initial conditions
        self.x0[0] = self.T[i+1]
        self.x0[1] = self.L[i+1]

        return self.u1[i+1],self.u2[i+1], self.T[i+1], self.L[i+1]

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

        self.lin_pc = linear_mpc(ω0_s,f0_c,Δω0_s,Δf0_c,noise)
        self.mpc_ws = 0
        self.mpc_fc = 0
        self.mpc_L = 0
        self.mpc_T = 0
        self.mpc_parts_per_iteration = 0
        self.mpc_total_parts = 0
        self.mpc_parts_on_spec = 0
        self.mpc_total_cost = 0
        self.mpc_total_revenue = 0
        self.mpc_total_profit = 0
        # efficiency
        self.mpc_efficiency = 0

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
        #Linear MPC
        self.mpc_ws, self.mpc_fc, self.mpc_T, self.mpc_L = self.lin_pc.step(self.cnt, self.mpc_T, self.mpc_L)

        # material density (kg / m^3)
        self.pho = 1.38 * 1000

        self.cnt += 1


        #MPC
        # material flow rate (meters^3 / second)
        self.mpc_Q = self.lin_pc.Q

        # mass flow rate in kg/s
        self.mpc_Q_m = self.mpc_Q * self.pho

        # degradation
        if self.mpc_T > 478:
            self.mpc_Q_m = self.mpc_Q_m * 0.9565
        elif self.mpc_T > 500:
            self.mpc_Q_m = self.mpc_Q_m * (-0.0036*self.mpc_T + 2.7565)

        self.mpc_parts_per_iteration = self.mpc_fc * Δt

        self.mpc_total_parts += self.mpc_parts_per_iteration

        # total cost ($)
        self.mpc_total_cost += self.cost * self.mpc_Q_m

        self.mpc_yield_ = 0

        if abs(self.mpc_L - self.L0) < self.ε:
            self.mpc_yield_ = self.mpc_fc
            self.mpc_parts_on_spec += self.mpc_yield_
            if self.mpc_parts_on_spec <= self.demand :
                # total revenue ($) assuming that we are selling all the good parts
                self.mpc_total_revenue += self.selling_price * self.mpc_Q_m
            
        # total profit ($) assuming that we are selling all the good parts
        self.mpc_total_profit = (self.mpc_total_revenue - self.mpc_total_cost)

        self.mpc_efficiency = float(self.mpc_parts_on_spec/self.mpc_total_parts)



    def episode_step(self) -> None:
        Δws = 0
        Δfc = 0
        
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
            "mpc_ws": float(self.mpc_ws),
            "mpc_fc": float(self.mpc_fc),
            "mpc_L": float(self.mpc_L),
            "mpc_T": float(self.mpc_T)
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
    file_name = 'state_space_data_mpc.csv'
    try:
        df_mpc = pd.read_csv(file_name)
    except:
        df_mpc = pd.DataFrame()
        
    onspec = 0
    # starting time
    start = tm.time()

    extruder_sim = ExtruderSimulation()

    ws0_list = [3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.5, 7.5, 7.5]
    fc0_list = [0.3,0.2,0.12,0.375,0.3,0.2, 0.1, 0.2, 0.325]
    demand_list = [75,75,75,120,120,120,190,190,190]
    i = 6 #demand scenario index
    
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
    mpc_ws_l = []
    mpc_fc_l = []
    mpc_T_l = []
    mpc_L_l = []
    mpc_profit_l = []
    model_name = f'linear_mpc_{ws0_list[i]}_{fc0_list[i]}_{demand_list[i]}'

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
        df_mpc = df_mpc.append(pd.DataFrame({'ws':[state["mpc_ws"]],'fc':[state["mpc_fc"]],'L':[state["mpc_L"]],
                                     'Q':[0], 'yield':[extruder_sim.mpc_yield_], 'T':[state['mpc_T']],
                                     'demand':[extruder_sim.demand], 'parts_on_spec':[extruder_sim.mpc_parts_on_spec],
                                     'efficiency':[extruder_sim.mpc_efficiency],'profit':[extruder_sim.mpc_total_profit],
                                     'time': [k], 'total_revenue': [extruder_sim.mpc_total_revenue], 'model_name':[model_name]
                                     }))
        

        mpc_profit_l.append(extruder_sim.mpc_total_profit)
        mpc_ws_l.append(extruder_sim.mpc_ws)
        mpc_fc_l.append(extruder_sim.mpc_fc)
        mpc_L_l.append(extruder_sim.mpc_L)
        mpc_T_l.append(extruder_sim.mpc_T)
        parts_on_spec_l.append(extruder_sim.mpc_parts_on_spec)
        eff_list.append(extruder_sim.mpc_efficiency)

        if graphics:
            plt.clf()
            
            plt.subplot(6,1,1)
            plt.plot([i for i in range(len(mpc_ws_l))],mpc_ws_l,'b.-',lw=2)
            plt.ylabel('Screw Angular Speed')
            plt.legend(['Screw Angular Speed'],loc='best')
            

            plt.subplot(7,1,2)
            plt.plot([i for i in range(len(mpc_fc_l))],mpc_fc_l,'b.-',lw=2)
            plt.ylabel('Cutter Frequency')
            plt.legend(['Cutter Frequency'],loc='best')
            
            plt.subplot(7,1,3)
            plt.plot([i for i in range(len(mpc_L_l))],mpc_L_l,'b.-',lw=2)
            plt.ylabel('Length')
            plt.legend(['Length'],loc='best')

            plt.subplot(7,1,4)
            plt.plot([i for i in range(len(mpc_T_l))],mpc_T_l,'b.-',lw=2)
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
            plt.plot([i for i in range(len(mpc_profit_l))],mpc_profit_l,'g.-',lw=1)
            #plt.plot([i for i in range(len(profit_l))],profit_l,'g.-',lw=1)
            plt.plot([i for i in range(len(profit_l))],[0 for i in range(len(profit_l))],'k-',lw=1)
            plt.ylabel('Profit $')
            plt.legend(['MPC Profit','Profit'],loc='best')

            
            plt.xlabel('Time (second)')
          

            plt.draw()
            plt.pause(0.001)

    print("Parts produced: ", extruder_sim.mpc_parts_on_spec)
    print("Maximum Temperature: ", max(T_l))
    print("Maximum Frequency: ", max(fc_l))
    print("Total Parts: ", extruder_sim.mpc_total_parts)
    print("Efficiency: ", extruder_sim.mpc_efficiency)
    print("Profit: ", extruder_sim.mpc_total_profit)
    # end time
    end = tm.time()
    time_run = end - start
    # total time taken
    print(f"Runtime of the program is {time_run} seconds")
    
    df_mpc.to_csv(file_name,index=False)


if __name__ == "__main__":
    main()
