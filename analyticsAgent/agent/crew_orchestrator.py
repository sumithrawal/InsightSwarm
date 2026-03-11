import os
from crewai import Agent, Task, Crew, Process
from .crew_tools import ingest_data_tool, analyze_data_tool, train_model_tool

def run_crew(filepath: str, prompt: str):
    print("\n INITIATING CLOUD SWARM CREW...")
    
    # Define Agents
    data_analyst = Agent(
        role='Senior Data Analyst',
        goal='Analyze datasets, uncover insights, and generate comprehensive EDA reports.',
        backstory='You are an expert data analyst with years of experience in Python, Pandas, and data visualization. You excel at finding hidden patterns in messy data and can determine the best target column from a dataset if none is specified.',
        verbose=True,
        allow_delegation=False,
        tools=[ingest_data_tool, analyze_data_tool]
    )

    ml_engineer = Agent(
        role='Machine Learning Engineer',
        goal='Build, evaluate, and deploy predictive models using automated pipelines.',
        backstory='You are a pragmatic ML engineer who knows how to choose the right algorithm for structured data tasks. You focus on robust validation and high performance. You wait for the data analyst to pass the name of the target variable before running model training.',
        verbose=True,
        allow_delegation=False,
        tools=[train_model_tool]
    )

    # Define Tasks
    analysis_task = Task(
        description=f"User request: '{prompt}'. The dataset is located at '{filepath}'. Use your tools to load the dataset, understand its schema, and run a full Exploratory Data Analysis. Figure out the best target column if the user hasn't explicitly specified one, and report back the key findings, most importantly the target column.",
        expected_output="A summary of the EDA report including dataset size, missing values, target column identified, and key insights.",
        agent=data_analyst
    )

    modeling_task = Task(
        description=f"Based on the analysis from the Data Analyst, note the selected target column and run the Model Trainer Tool on '{filepath}'. Ensure that you correctly specify the target variable when calling the tool.",
        expected_output="A summary of the ML training process highlighting the best model, its performance metrics, and where it is saved.",
        agent=ml_engineer
    )

    # Form the Crew
    swarm_crew = Crew(
        agents=[data_analyst, ml_engineer],
        tasks=[analysis_task, modeling_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute
    print("\n Crew is starting its work based on your prompt...")
    result = swarm_crew.kickoff()
    
    print("\n" + "="*50)
    print(" CREW EXECUTION FINISHED")
    print("="*50)
    print(result)
    return result
