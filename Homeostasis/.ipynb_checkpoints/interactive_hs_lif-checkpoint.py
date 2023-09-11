import torch
from pymonntorch import Behavior, SynapseGroup, Network, NeuronGroup, Recorder, EventRecorder
import matplotlib.pyplot as plt
from matplotlib.pyplot import  Button, Slider
import numpy as np

class ActivityBaseHomeostasis(Behavior):
    def initialize(self, neurons):
        activity_rate = self.parameter("activity_rate", 5)
        self.window_size = self.parameter("window_size", 50)
        self.updating_rate = self.parameter("updating_rate", 8)
        self.decay_rate = self.parameter("decay_rate", 0.9)

        self.firing_reward = 1
        self.non_firing_penalty = -activity_rate / (self.window_size - activity_rate)

        self.activities = neurons.vector(mode="zeros")

    def forward(self, neurons):
        add_activitiies = torch.where(
            neurons.spikes, self.firing_reward, self.non_firing_penalty
        )

        self.activities += add_activitiies

        if (neurons.iteration % self.window_size) == 0:
            change = -self.activities * self.updating_rate
            neurons.threshold -= change
            self.activities.fill_(0)
            self.updating_rate *= self.decay_rate


class Real_time_activity_base_Homeostasis(Behavior):

    def initialize(self, neurons):
        activity_rate = self.parameter("activity_rate", 5)
        self.window_size = self.parameter("window_size", 50)
        self.updating_rate = self.parameter("updating_rate", 8)
        self.decay_rate = self.parameter("decay_rate", 0.9)

        self.firing_reward = 1
        self.non_firing_penalty = -activity_rate / (self.window_size - activity_rate)

        self.activities = neurons.vector(mode="zeros")

    def forward(self, neurons):
        add_activitiies = torch.where(
            neurons.spikes, self.firing_reward, self.non_firing_penalty
        )

        self.activities += add_activitiies

        # if (neurons.iteration % self.window_size) == 0:
        change = -self.activities * self.updating_rate
        neurons.threshold -= change
        self.activities.fill_(0)
        self.updating_rate *= self.decay_rate


class Voltage_base_Homeostasis(Behavior):

    def initialize(self, neurons):
        self.add_tag('Voltage_base_Homeostasis')

        target_act = self.parameter('target_voltage', 0.05, neurons)
        self.max_ta = self.parameter('max_ta', target_act, neurons)
        self.min_ta = self.parameter('min_ta', target_act, neurons)
        self.adj_strength = -self.parameter('eta_ip', 0.001, neurons)

        neurons.exhaustion = neurons.vector()

    def forward(self, neurons):
        greater = ((neurons.v > self.max_ta) * -1).type(torch.float32)
        smaller = ((neurons.v < self.min_ta) * 1).type(torch.float32)

        greater *= neurons.v - self.max_ta
        smaller *= self.min_ta - neurons.v

        change = (greater + smaller) * self.adj_strength
        neurons.exhaustion += change
        neurons.v -= neurons.exhaustion

class LIF(Behavior):
    
    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag("LIF")
        self.set_parameters_as_variables(neurons)
        neurons.v = neurons.vector("uniform") * (neurons.threshold - neurons.v_reset) * 1.1
        neurons.v += neurons.v_reset
        neurons.spikes = neurons.v > neurons.threshold
        
    def dv_dt(self, neurons):
        leakage = -(neurons.v - neurons.v_rest)
        return ((leakage + (neurons.R * neurons.I)) / neurons.tau) * neurons.dt
        
    def forward(self, neurons):
        neurons.v += self.dv_dt(neurons)
        neurons.v[neurons.spikes] = neurons.v_reset
        neurons.spikes = neurons.v >= neurons.threshold

class LIF_INPUT(Behavior):
    
    def initialize(self, synapse):
        self.coef = self.parameter("coef", None)
        self.density = self.parameter("density", None)
        torch.manual_seed(42)
        self.W = synapse.matrix(mode='uniform')
        # print(self.W.shape)

    def forward(self, synapse):
        spikes = synapse.src.spikes
        synapse.I = torch.sum(self.W[spikes], axis=0) * self.coef

class Dendrite(Behavior):              

    def forward(self, neurons):
        for synapse in neurons.afferent_synapses['GLUTAMATE']: 
            neurons.I += synapse.I
            
        # for synapse in neurons.afferent_synapses['GABA']:
        #     neurons.I -= synapse.I

class Current(Behavior):
    def initialize(self, neurons):
        self.current = self.parameter("current", None)
        self.inc_current = self.parameter("inc_current", None)
        neurons.I = neurons.vector(self.current)

    def forward(self, neurons):
        # neurons.I = neurons.vector(self.current)
        neurons.I += self.inc_current



def stream_izh_voltage(max_ta=-20, min_ta=-62, eta_ip=0.5, current = 2, 
                inc_currentt=0.5, R=5, tau=10, v_rest=-67, v_reset=-75, threshold=-37, coef=0.5, iter=200):

    ITER = iter
    EXC_SIZE = 1

    pop_params = {
        "v_reset" : v_reset,
        "dt" : 1,
        "v_rest": v_rest,
        "tau" : tau,
        "R" : R,
        "threshold" : threshold,
    }
    my_net = Network()
    ng_exc = NeuronGroup(
        net=my_net,
        size=EXC_SIZE,
        tag="exc_neurons",
       behavior={
            4: Current(current = current, inc_current = inc_currentt),
            5: Dendrite(),
            7: LIF(**pop_params),
            6: Voltage_base_Homeostasis(target_voltage = -40, max_ta = max_ta, min_ta = min_ta, eta_ip = eta_ip),
            9: Recorder(["n.v", "torch.mean(n.v)","n.I"], auto_annotate=False),
            11: EventRecorder(["spikes"]),
        },
    )

    SynapseGroup(src=ng_exc, dst=ng_exc, net=my_net, tag="GLUTAMATE", behavior={3: LIF_INPUT(coef=coef, density=0.1)})

    my_net.initialize()
    my_net.simulate_iterations(ITER)
    
    return my_net["n.v",0][:, :1]

def stream_izh_activity(window_size=50, update_rate=0.5, activity_rate=5, decay_rate=0.9, current = 2, 
                inc_currentt=0.5, R=5, tau=10, v_rest=-67, v_reset=-75, threshold=-37, coef=0.5, iter=200):

    ITER = iter
    EXC_SIZE = 1

    pop_params = {
        "v_reset" : v_reset,
        "dt" : 1,
        "v_rest": v_rest,
        "tau" : tau,
        "R" : R,
        "threshold" : threshold,
    }
    my_net = Network()
    ng_exc = NeuronGroup(
        net=my_net,
        size=EXC_SIZE,
        tag="exc_neurons",
       behavior={
            4: Current(current = current, inc_current = inc_currentt),
            5: Dendrite(),
            7: LIF(**pop_params),
            6: ActivityBaseHomeostasis(window_size=window_size, updating_rate=update_rate, 
                activity_rate=activity_rate, decay_rate=decay_rate),
            9: Recorder(["n.v", "torch.mean(n.v)","n.I"], auto_annotate=False),
            11: EventRecorder(["spikes"]),
        },
    )

    SynapseGroup(src=ng_exc, dst=ng_exc, net=my_net, tag="GLUTAMATE", behavior={3: LIF_INPUT(coef=coef, density=0.1)})

    my_net.initialize()
    my_net.simulate_iterations(ITER)
    
    return my_net["n.v",0][:, :1]


def interactive_lif(mode = 'voltage_base_homeostasis', iter_init=200):  

    if mode == 'voltage_base_homeostasis':
        max_ta=-20 
        min_ta=-62 
        eta_ip=0.5 
        current = 7
        inc_current=0.5
        R=5
        tau=10
        v_rest=-67
        v_reset=-75
        threshold=-37
        coef = 0.5

        axis_color = 'lightgoldenrodyellow'

        fig = plt.figure("LIF Neuron", figsize=(16, 10))
        ax = fig.add_subplot(111)
        plt.title("Interactive LIF Neuron With Homeostasis")
        fig.subplots_adjust(left=0.1, bottom=0.5)
        # print(net["n.v",0][:, :1].shape)
        line = plt.plot(torch.ones((iter_init))*1., label="Membrane Potential")[0]
        # line2 = plt.plot(torch.ones((iter_init))*1., label="Applied Current")[0]
        line3 = plt.plot(torch.ones((iter_init))* -37, label="Threshold Voltage")[0]
        plt.ylim([-100, 220])

        plt.legend(loc="upper right")

        plt.ylabel("Potential [V]")
        plt.xlabel("Time [s]")


        max_ta_axis = plt.axes([0.1, 0.40, 0.65, 0.03], facecolor=axis_color)
        max_ta_slider = Slider(max_ta_axis, '$max-ta$', -100, 100, valinit=max_ta)

        min_ta_axis = plt.axes([0.1, 0.37, 0.65, 0.03], facecolor=axis_color)
        min_ta_slider = Slider(min_ta_axis, '$min-ta$', -100, 100, valinit=min_ta)

        eta_ip_axis = plt.axes([0.1, 0.34, 0.65, 0.03], facecolor=axis_color)
        eta_ip_slider = Slider(eta_ip_axis, '$eta-ip$', 0.0, 5, valinit=eta_ip)

        current_axis = plt.axes([0.1, 0.31, 0.65, 0.03], facecolor=axis_color)
        current_slider = Slider(current_axis, '$current$', 0, 20, valinit=current)

        inc_current_axis = plt.axes([0.1, 0.28, 0.65, 0.03], facecolor=axis_color)
        inc_current_slider = Slider(inc_current_axis, '$inc-current$', 0.001, 10, valinit=inc_current)

        R_axis = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axis_color)
        R_slider = Slider(R_axis, '$R$', 0, 10, valinit=R)

        tau_axis = plt.axes([0.1, 0.22, 0.65, 0.03], facecolor=axis_color)
        tau_slider = Slider(tau_axis, '$tau$', 5, 20, valinit=tau)

        v_rest_axis = plt.axes([0.1, 0.19, 0.65, 0.03], facecolor=axis_color)
        v_rest_slider = Slider(v_rest_axis, '$v-rest$', -100, -10, valinit=v_rest)

        v_reset_axis = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor=axis_color)
        v_reset_slider = Slider(v_reset_axis, '$v-reset$', -100, -10, valinit=v_reset)

        th_axis = plt.axes([0.1, 0.13, 0.65, 0.03], facecolor=axis_color)
        th_slider = Slider(th_axis, '$threshold$', -50, 0, valinit=threshold)

        coef_axis = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axis_color)
        coef_slider = Slider(coef_axis, '$coef$', 0, 20, valinit=coef)

        def update(val):
            v= stream_izh_voltage(max_ta=max_ta_slider.val, min_ta=min_ta_slider.val, eta_ip=eta_ip_slider.val, current = current_slider.val, 
                    inc_currentt=inc_current_slider.val, R=R_slider.val, tau=tau_slider.val, v_rest=v_rest_slider.val, v_reset=v_reset_slider.val,
                    threshold=th_slider.val, coef=coef_slider.val, iter=iter_init)
            line.set_ydata(v)
            line3.set_ydata(th_slider.val)

        max_ta_slider.on_changed(update)
        min_ta_slider.on_changed(update)
        eta_ip_slider.on_changed(update)
        current_slider.on_changed(update)
        inc_current_slider.on_changed(update)
        R_slider.on_changed(update)
        tau_slider.on_changed(update)
        v_rest_slider.on_changed(update)
        v_reset_slider.on_changed(update)
        th_slider.on_changed(update)
        coef_slider.on_changed(update)

        plt.show()

    elif mode == 'activity_base_homeostasis':
        window_size=50
        updating_rate = 0.5
        activity_rate=5
        decay_rate=0.9
        current = 7
        inc_current=0.5
        R=5
        tau=10
        v_rest=-67
        v_reset=-75
        threshold=-37
        coef = 0.5

        axis_color = 'lightgoldenrodyellow'

        fig = plt.figure("LIF Neuron", figsize=(16, 10))
        ax = fig.add_subplot(111)
        plt.title("Interactive LIF Neuron With Homeostasis")
        fig.subplots_adjust(left=0.1, bottom=0.5)
        # print(net["n.v",0][:, :1].shape)
        line = plt.plot(torch.ones((iter_init))*1., label="Membrane Potential")[0]
        # line2 = plt.plot(torch.ones((iter_init))*1., label="Applied Current")[0]
        line3 = plt.plot(torch.ones((iter_init))* -37, label="Threshold Voltage")[0]
        plt.ylim([-100, 220])

        plt.legend(loc="upper right")

        plt.ylabel("Potential [V]")
        plt.xlabel("Time [s]")


        window_size_axis = plt.axes([0.1, 0.40, 0.65, 0.03], facecolor=axis_color)
        window_slider = Slider(window_size_axis, '$window-size$', 0, 500, valstep=1, valinit=window_size)

        update_rate_axis = plt.axes([0.1, 0.37, 0.65, 0.03], facecolor=axis_color)
        update_rate_slider = Slider(update_rate_axis, '$update-rate$', 0, 50, valinit=updating_rate)

        activity_axis = plt.axes([0.1, 0.34, 0.65, 0.03], facecolor=axis_color)
        activity_slider = Slider(activity_axis, '$activity-rate$', 0.0, 30, valinit=activity_rate)

        decay_axis = plt.axes([0.1, 0.31, 0.65, 0.03], facecolor=axis_color)
        decay_slider = Slider(decay_axis, '$decay-rate$', 0.0, 5, valinit=decay_rate)

        current_axis = plt.axes([0.1, 0.28, 0.65, 0.03], facecolor=axis_color)
        current_slider = Slider(current_axis, '$current$', 0, 20, valinit=current)

        inc_current_axis = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axis_color)
        inc_current_slider = Slider(inc_current_axis, '$inc-current$', 0.001, 10, valinit=inc_current)

        R_axis = plt.axes([0.1, 0.22, 0.65, 0.03], facecolor=axis_color)
        R_slider = Slider(R_axis, '$R$', 0, 10, valinit=R)

        tau_axis = plt.axes([0.1, 0.19, 0.65, 0.03], facecolor=axis_color)
        tau_slider = Slider(tau_axis, '$tau$', 5, 20, valinit=tau)

        v_rest_axis = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor=axis_color)
        v_rest_slider = Slider(v_rest_axis, '$v-rest$', -100, -10, valinit=v_rest)

        v_reset_axis = plt.axes([0.1, 0.13, 0.65, 0.03], facecolor=axis_color)
        v_reset_slider = Slider(v_reset_axis, '$v-reset$', -100, -10, valinit=v_reset)

        th_axis = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axis_color)
        th_slider = Slider(th_axis, '$threshold$', -50, 0, valinit=threshold)

        coef_axis = plt.axes([0.1, 0.08, 0.65, 0.03], facecolor=axis_color)
        coef_slider = Slider(coef_axis, '$coef$', 0, 20, valinit=coef)

        def update(val):

            v = stream_izh_activity(window_size=window_slider.val, update_rate=update_rate_slider.val,
             activity_rate=activity_slider.val, decay_rate=decay_slider.val,
                     current = current_slider.val, inc_currentt=inc_current_slider.val, R=R_slider.val, tau=tau_slider.val,
                      v_rest=v_rest_slider.val, v_reset=v_reset_slider.val,
                    threshold=th_slider.val, coef=coef_slider.val, iter=iter_init)

            line.set_ydata(v)
            line3.set_ydata(th_slider.val)

        window_slider.on_changed(update)
        update_rate_slider.on_changed(update)
        activity_slider.on_changed(update)
        decay_slider.on_changed(update)
        current_slider.on_changed(update)
        inc_current_slider.on_changed(update)
        R_slider.on_changed(update)
        tau_slider.on_changed(update)
        v_rest_slider.on_changed(update)
        v_reset_slider.on_changed(update)
        th_slider.on_changed(update)
        coef_slider.on_changed(update)

        plt.show()

iterations = 200
inp = int(input("choose your Homeistasis mode:\n 1. voltage base\n 2. activity base\n"))
if inp == 1:
    interactive_lif('voltage_base_homeostasis', iterations)
elif inp == 2:
    interactive_lif('activity_base_homeostasis', iterations)