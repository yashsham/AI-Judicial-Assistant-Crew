# agents/constitutional_query_agent.py

from crewai import Agent, Task

class ConstitutionalQueryAgents:
    def make_query_agent(self, llm, tools):
        return Agent(
            role='Constitutional Law Specialist AI',
            goal='Identify potential constitutional questions in a case summary and retrieve relevant articles from the Constitution of India using the provided RAG tool.',
            backstory=(
                "You are an AI expert in Indian Constitutional Law. Your function is to analyze "
                "the facts of a case and pinpoint specific articles, clauses, and amendments from "
                "the Constitution of India that are pertinent to the legal questions at hand."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools # This will be our Constitution RAG tool
        )

    def make_constitutional_analysis_task(self, agent, context):
        return Task(
            description=(
                "Based on the provided case summary, identify any and all potential constitutional "
                "issues. For each issue, use the 'Constitution RAG Tool' to search the Constitution of India "
                "and find the most relevant articles or sections. List the issues and the corresponding "
                "constitutional provisions you find."
            ),
            expected_output=(
                "A structured list of potential constitutional issues. Each issue should be followed by the "
                "full text of the relevant article(s) or clause(s) retrieved from the Constitution."
            ),
            agent=agent,
            context=context # This receives the summary from the first agent
        )