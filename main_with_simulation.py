import math
# from mido import Message
# import mido
import time
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# import pandas as pd
import sys
import os
import csv
from datetime import datetime

############ change it to your name
tester_name = 'Simulator'

##### Hyper Parameters
DURATION = 0.3
PAUSE = 1
RND_RANGE = [2, 4, 6, 8, 10, 12]  # 8 or 4 or something
TEST_ROUND = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # 10 or 20 or 50 or ...
TIMES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
REPEAT = False


############

class Tone:

    def __init__(self, pitch, vel):
        self.pitch = pitch
        self.vel = vel

    def change_velocity(self, new_vel):
        self.vel = new_vel


class SoundPlayer:

    def __init__(self, duration, pause):
        self.duration = duration
        self.pause = pause

    def play_pair(self, ref_tone: Tone, var_tone: Tone):
        pause = self.pause
        # time.sleep(1)
        idx = 1
        return idx


class TestManager:

    def __init__(self, xiang_pitch, xiang, var_pitch, vel_min, vel_max):
        self.xiang = xiang
        self.var_pitch = var_pitch
        self.vel_list = np.linspace(vel_min, vel_max, vel_max - vel_min + 1)

        self.test_vel = []
        self.test_fb = []

        # based on the definition of xiang
        self.ref_stimuli = Tone(pitch=xiang_pitch, vel=xiang)
        self.sound_player = SoundPlayer(DURATION, PAUSE)

    def play_round(self, test_round, randrange):
        if test_round == 1:
            vel_max = self.xiang + 3 * 8
            is_var_louder = self.run_test(var_vel=vel_max)

            while not is_var_louder and vel_max <= 90:
                vel_max += 5
                is_var_louder = self.run_test(var_vel=vel_max)
            if not is_var_louder:
                print(
                    f"No point of subject equality can be found for pitch {self.var_pitch} of {self.xiang} xiang")
                sys.exit(0)
        elif test_round == 2:
            vel_min = self.xiang - 3 * 8
            is_var_louder = self.run_test(var_vel=vel_min)

            while is_var_louder and vel_min >= 20:
                vel_min -= 5
                is_var_louder = self.run_test(var_vel=vel_min)
            if is_var_louder:
                print(
                    f"No point of subject equality can be found for pitch {self.var_pitch} of {self.xiang} xiang")
                sys.exit(0)
        else:
            # rnd_range = int(randrange / 2) if test_round >= 5 else randrange
            vel = self.get_var_velocity()[0]
            ## used to be 16
            rand = random.randrange(-randrange, randrange, 1)
            vel += rand

            # print(f'vel: {vel}, rand: {rand}')

            is_var_louder = self.run_test(var_vel=int(vel))
        return is_var_louder

    def simulator(self, var_v):
        diff = self.ref_stimuli.vel - var_v.vel
        mean = 58
        prob = 0
        # prob = 3.5 * stats.norm.pdf(var_v.vel, mean, 2.8)  # Gaussian Curve  50-56 57-59 59-65
        if var_v.vel < 50 or var_v.vel > 65:
            prob = 0 + round(random.uniform(0, 0.05), 3)
        elif 50 <= var_v.vel < 57:
            prob = 0.071 * var_v.vel - 3.571
        elif 57 <= var_v.vel <= 59:
            prob = 0.5
        else:
            prob = -0.071 * var_v.vel + 4.714

        if prob < 0:
            prob = 0
        if diff >= 0:
            return np.random.choice(2, 1, p=[prob, 1 - prob])[0]
        else:
            return np.random.choice(2, 1, p=[1 - prob, prob])[0]

    def run_test(self, var_vel):
        # handle corner case
        if var_vel < 30:
            var_vel = 30
        if var_vel > 90:
            var_vel = 90

        var_stimuli = Tone(pitch=self.var_pitch, vel=var_vel)
        ref_idx = self.sound_player.play_pair(self.ref_stimuli, var_stimuli)

        # print("Please enter O if the former is louder, enter P if the latter is louder:")
        # feedback = input()
        feedback = self.simulator(var_stimuli)

        # print(feedback)
        # classification, 0 for ref is louder, 1 for var is louder
        if int(feedback) == ref_idx:
            is_var_louder = 0
        else:
            is_var_louder = 1

        self.test_vel.append([var_vel])
        self.test_fb.append(is_var_louder)

        # Write a pair of > or <
        # with open(r'simulator_files/all_pairs_simulator.csv', 'a') as all_pairs:
        #     writer = csv.writer(all_pairs)
        #     if not ref_idx:
        #         writer.writerow([tester_name, self.ref_stimuli.pitch, self.ref_stimuli.vel,
        #                          self.var_pitch, var_vel, is_var_louder ^ ref_idx, ref_idx, DURATION, PAUSE, REPEAT,
        #                          datetime.now()])
        #     else:
        #         writer.writerow([tester_name, self.var_pitch, var_vel, self.ref_stimuli.pitch, self.ref_stimuli.vel,
        #                          is_var_louder ^ ref_idx, ref_idx, DURATION, PAUSE, REPEAT, datetime.now()])
        return is_var_louder

    def get_var_velocity(self):
        # fit curve
        lgc = LogisticRegression(random_state=0).fit(
            self.test_vel, self.test_fb)

        prob_all = lgc.predict_proba(
            self.vel_list.reshape(len(self.vel_list), 1))

        # probability of var is louder than ref
        prob_var = np.array(list(map(lambda p: p[1], prob_all)))

        relative_prob_var = abs(prob_var - 0.5)
        pse_idx = np.where(relative_prob_var == min(relative_prob_var))[0][0]

        pse = self.vel_list[pse_idx]

        s_ave = 0
        for v in self.test_vel:
            p = prob_var[v[0] - 30]  # p need to p > 1
            try:
                s = (pse - v) / math.log(1 / p - 1, math.e)
            except:
                pass
            else:
                s_ave += s

        s_ave = s_ave / len(self.test_vel)
        sd = (s_ave * math.pi) / math.sqrt(3)  # standard deviation

        # self.draw_plot(prob_var, pse, pse_idx)

        return pse, s, sd

    def draw_plot(self, prob, pse, pse_idx):
        plt.title(
            f'Fitted Logistic function for round {len(self.test_vel) - 2}')
        plt.xlabel("midi velocity")
        plt.ylabel("Prob that var is louder than ref")

        x = self.vel_list
        y = prob

        x_var = []
        y_var = []
        x_ref = []
        y_ref = []
        for idx, v in enumerate(self.test_vel):
            if self.test_fb[idx]:
                x_var.append(v)
                y_var.append(y[int(v - min(x))])
            else:
                x_ref.append(v)
                y_ref.append(y[int(v - min(x))])

        plt.scatter(x_var, y_var, marker='o', label="var is louder")
        plt.scatter(x_ref, y_ref, marker='^', label="ref is louder")
        plt.scatter(pse, y[pse_idx], marker='*', c='r', label="estimated PSE")

        plt.plot(x, y)

        plt.legend(loc='upper left')
        plt.show()


def key_map(key):
    M = {'o': '0', 'p': '1'}
    return M.get(key)


def main(var_pitch, ref_vel, ref_pitch=69, total_test_round=10, rand_range=4):
    # set xiang
    xiang = ref_vel

    test_manager = TestManager(
        xiang_pitch=ref_pitch,
        xiang=xiang,
        var_pitch=var_pitch,
        vel_min=30,
        vel_max=90,
    )
    test_round = 0
    while test_round < total_test_round + 2:
        test_round += 1
        test_manager.play_round(test_round, rand_range)
    pse, s, sd = test_manager.get_var_velocity()

    # print(
    #     f"The point of subjective equality for pitch {var_pitch} of {xiang} xiang is: {pse}, sd is: {s}")
    return int(pse)


#############
# Before Starting the test, determine how many XIANG levels i.e. 需要测几条线
# for each 线, we need to add a new csv document in this folder
# suggested naming: xiang_xx_tester_xxxxx_pse_data.csv

### Data used for training the ML model
if __name__ == '__main__':

    # ref_pitch = int(input('Enter reference pitch (default is 69/A4): '))
    ref_pitch = 69
    # ref_vel = int(input('Enter reference velocity: '))
    ref_vel = 60
    # var_pitch = int(input('Enter variable pitch: '))
    var_pitch = 63
    for t in range(len(TEST_ROUND)):
        for r in RND_RANGE:
            for time in TIMES:
                if not os.path.exists(f'simulator_files/pse_{r}_simulator{TEST_ROUND[t]}_{time}.csv'):
                    pse_records_f = open(f'simulator_files/pse_{r}_simulator{TEST_ROUND[t]}_{time}.csv', 'w')
                    pse_records_f.write(
                        'tester,ref_p,ref_v,var_p,var_pse_v,duration,pause,repeat,rnd_range,test_round,simulation_round,timestamp\n')
                    pse_records_f.close()
                with open(f'simulator_files/pse_{r}_simulator{TEST_ROUND[t]}_{time}.csv', 'a') as pse_records:
                    writer = csv.writer(pse_records)
                    i = 0
                    while i < time:
                        try:
                            pse = main(var_pitch, ref_vel, ref_pitch=ref_pitch, total_test_round=TEST_ROUND[t],
                                       rand_range=r)
                        except:
                            continue
                        writer.writerow(
                            [tester_name, ref_pitch, ref_vel, var_pitch, pse, DURATION, PAUSE, REPEAT, RND_RANGE,
                             TEST_ROUND, time,
                             datetime.now()])
                        if i % (time // 5) == 0:
                            print('Test Round', TEST_ROUND[t], 'RandRange:', r, int(i / time * 100), "%")
                        i += 1
