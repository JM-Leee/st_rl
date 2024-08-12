import streamlit as st
import numpy as np
import pickle
import plotly.graph_objs as go
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import Env, spaces
from stable_baselines3.common.callbacks import BaseCallback

# Custom environment class
class ModifiedQuadraticEnv(Env):
    def __init__(self):
        super(ModifiedQuadraticEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([201, 201])  # -100부터 100까지의 정수 값
        self.observation_space = spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)
        self.state = np.random.uniform(low=-100, high=100, size=(2,))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-100, high=100, size=(2,))  # 초기 상태를 -100에서 100 사이로 설정
        return self.state, {}

    def step(self, action):
        x_action, y_action = action
        x_action -= 100
        y_action -= 100
        self.state = np.array([x_action, y_action], dtype=np.float32)
        x, y = self.state
        z = 2 * x**2 + y**2 + 100
        reward = -z / 10  # z값이 작아질수록 보상이 커짐
        done = False
        if z < 101:  # 충분히 작은 z 값에 도달하면 종료
            done = True
            reward += 1000  # 보상을 추가하여 목표 상태 도달을 강화
        info = {}
        return self.state, reward, done, False, info

    def render(self, mode='human'):
        x, y = self.state
        print(f"State: x={x}, y={y}, z={2*x**2 + y**2 + 100}")

# Streamlit app

st.title("Reinforcement Learning with PPO")

if 'model' not in st.session_state:
    st.session_state.model = None

if 'training_data' not in st.session_state:
    st.session_state.training_data = None

env = ModifiedQuadraticEnv()

# Train button
if st.button('Start Training'):
    with st.spinner('Training the model...'):
        model = PPO('MlpPolicy', env, verbose=1)
        st.session_state.model = model

        class ZValueCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(ZValueCallback, self).__init__(verbose)
                self.episode_z_values = []
                self.states_visited = []

            def _on_step(self) -> bool:
                x, y = self.locals['new_obs'][0]
                z = 2 * x**2 + y**2 + 100
                self.episode_z_values.append(z)
                self.states_visited.append((x, y))
                return True

            def _on_training_end(self) -> None:
                st.session_state.training_data = {
                    "episode_z_values": self.episode_z_values,
                    "states_visited": self.states_visited
                }
                with open("training_data_streamlit.pkl", "wb") as f:
                    pickle.dump(st.session_state.training_data, f)
                st.success("Training data saved.")
        
        z_value_callback = ZValueCallback()
        model.learn(total_timesteps=50000, callback=z_value_callback)
        model.save("ppo_modified_quadratic_streamlit")
        st.success("Model training completed and saved!")

# Show results if available
if st.session_state.training_data:
    training_data = st.session_state.training_data
    episode_z_values = training_data["episode_z_values"]
    states_visited = np.array(training_data["states_visited"])

    # Plot z value graph
    z_value_fig = go.Figure(data=go.Scatter(y=episode_z_values, mode='lines'))
    z_value_fig.update_layout(title="Z Value per Step During Training", xaxis_title="Step", yaxis_title="Z Value")
    st.plotly_chart(z_value_fig)

    # # Plot path graph
    # path_fig = go.Figure(data=go.Scatter(x=states_visited[:, 0], y=states_visited[:, 1], mode='lines+markers'))
    # path_fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', size=10), name='Target'))
    # path_fig.update_layout(title="States Visited Over Time", xaxis_title="X", yaxis_title="Y")
    # st.plotly_chart(path_fig)

    final_state = states_visited[-1]
    x, y = final_state
    z = 2 * x**2 + y**2 + 100
    st.write(f"Final state: x={x}, y={y}, z={z}")

# Load model button
if st.button('Load Model'):
    model = PPO.load("ppo_modified_quadratic_streamlit")
    st.session_state.model = model
    st.success("Model loaded successfully!")
