# MATEA: A Framework for Analyzing the Emergence of Intention through Thoughts in Large Language Model-based Agent Simulation

This repository is associated with our research paper titled "MATEA: A Framework for Analyzing the Emergence of Intention through Thoughts in Large Language Model-based Agent Simulation".We presents the Multi-Agent Thought Emergence Analysis Framework (MATEA), a hierarchical architecture that connects individual agent thoughts with macro-level social emergence. MATEA uses Inspector Agents to monitor individual agent thoughts and Analysis Agents to detect changes and generate new thoughts. This repository includes models we use in our research, provide a simulation program for the food delivery city and related visual emergence analysis code.

## Setting Up the Environment

## Step 1. Prepare

To set up your environment, you need run the simulation program under Linux system.we tested our environment on Ubuntu 22.04 LTS.
You can choose to deploy the large model DeepSeek R1 locally or call the OpenAI API interface.
If you choose to deploy the large model DeepSeek R1 locally, please modify the local model path in the file models/agents/LLMAgent.py.
If you choose to call the OpenAI API interface
you need get your OpenAI API key. You can fill in your OPENAI_API_KEY in models/agents/LLMAgent.py.

```
os.environ["AZURE_OPENAI_API_KEY"] = "Your openai api"
os.environ["AZURE_OPENAI_ENDPOINT"] = "your ENDPOINT"
```

<br/>

### Step 2. Install requirements.txt

Install everything listed in the requirements.txt file (I strongly recommend first setting up a virtualenv as usual). A note on Python version: we tested our environment on Python 3.10.0

## Running a Simulation

To run a new simulation, You can directly execute the following code in the SocialInvolution/entity/ directory.

```
python city.py
```

You can modify the relevant experimental configuration directly in city.py. The default configuration is 100 agents to simulate and run 3600 steps. The actual running time is more than 6 hours.

After the run is complete, you can run any visualization code in the SocialInvolution/algorithm/drow_echarts directory and get the corresponding emergence analysis results.
