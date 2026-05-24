from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel
from main import IKAgent

app = FastAPI()

class GoalInput(BaseModel):
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class LearnInput(BaseModel):
    goal: list
    q_agent: list
    q_actual: list
    ee_actual: list


with open("model.pkl", "rb") as f:
    nn = pickle.load(f)

agent = IKAgent(nn)
latest_goal = None
# safety: ensure these exist
X_mean = getattr(nn, "X_mean", np.zeros(6))
X_std = getattr(nn, "X_std", np.ones(6))
Y_mean = getattr(nn, "Y_mean", np.zeros(6))
Y_std = getattr(nn, "Y_std", np.ones(6))

@app.post("/predict")
def predict(data: GoalInput):

    global latest_goal

    goal = np.array([
        data.x,
        data.y,
        data.z,
        data.roll,
        data.pitch,
        data.yaw
    ])

    latest_goal = goal  

    q = agent.act(goal)

    return {"joints": q.tolist()}

@app.get("/goal")
def get_goal():
    global latest_goal

    if latest_goal is None:
        return {"goal": []}
    return {
        "goal": latest_goal.astype(float).tolist()
    }

@app.post("/learn")
def learn(data: LearnInput):

    goal = np.array(data.goal, dtype=np.float32)
    q_agent = np.array(data.q_agent, dtype=np.float32)
    q_actual = np.array(data.q_actual, dtype=np.float32)
    ee_actual = np.array(data.ee_actual, dtype=np.float32)

    if goal.shape != (6,) or q_agent.shape != (6,) or q_actual.shape != (6,):
        return {"status": "shape_error"}

    error = np.linalg.norm(goal[:3] - ee_actual)
    smoothness = np.linalg.norm(q_agent - agent.prev_q)
    energy = np.sum(q_agent ** 2)

    reward = -(1.0 * error + 0.3 * smoothness + 0.1 * energy)

    goal_norm = (goal - X_mean) / (X_std + 1e-8)
    q_norm = (q_actual - Y_mean) / (Y_std + 1e-8)

    pred = nn.forward(goal_norm.reshape(1, -1))
    nn.backward(goal_norm.reshape(1, -1), q_norm.reshape(1, -1), pred)

    return {
        "status": "updated",
        "reward": float(reward)
    }
