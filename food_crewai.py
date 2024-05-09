from crewai import Crew
from crewai.process import Process
import os
from langchain_groq import ChatGroq

from src.food_agents import ChefAgents
from src.food_task import ChefTask



agents = ChefAgents()
tasks = ChefTask()

LLM = os.getenv('LLM')
API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(api_key=API_KEY, model=LLM)

chef_agent = agents.chef_agent()
nutrition_agent = agents.nutrition_agent()
ingridient_agent = agents.ingridient_agent()
image_visualizer_agent = agents.image_visualizer_agent()
grocery_agent = agents.grocery_agent()
wine_agent = agents.wine_agent()
trivia_agent = agents.trivia_agent()
manager_agent = agents.manager_agent()



if __name__ == "__main__":

    print("\n\t==== Welcome to Food World =====",end="\n")

    input = input("\tHow can I help you... ")

        
    task1 = tasks.chef_task(
        agent=manager_agent,
        input=input,
        )


    crew = Crew(
    agents=[chef_agent,
            nutrition_agent,
            ingridient_agent,
            image_visualizer_agent,
            grocery_agent,
            wine_agent,
            trivia_agent,
            manager_agent
            ],
    tasks=[
            task1
        # tasks.crew_task(input='how to make chicken biryani'),
        
        # tasks.ingriedient_finder
    ],
    # function_calling_llm=function_llm,
    # full_output=True,
    verbose=1,
    process=Process.hierarchical,
    # manager_callbacks=manager_agent,
    manager_llm=llm
)

    result = crew.kickoff()
    print("\n\t++++++++++++++++++++++++++++++\n")
    print(result)