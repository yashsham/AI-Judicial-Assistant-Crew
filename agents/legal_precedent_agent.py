# agents/legal_precedent_agent.py

from crewai import Agent, Task
from crewai.tools import tool

# We will define a simple tool directly here for demonstration
@tool("Legal Precedent Search Tool")
def legal_precedent_search_tool(query: str) -> str:
    """
    Simulates searching a legal database for past judgments related to a query.
    In a real application, this would connect to a real legal database API.
    - query: A description of the legal issue to search for.
    """
    # This is a placeholder response
    return (
        "Simulated Search Result:\n"
        "1. Kesavananda Bharati v. State of Kerala (1973): Established the basic structure doctrine of the Indian Constitution.\n"
        "2. Maneka Gandhi v. Union of India (1978): Expanded the interpretation of 'right to life' under Article 21.\n"
        "Please note: These are simulated results and a real implementation would yield specific, relevant case law based on the query."
    )

class LegalPrecedentAgents:
    def make_precedent_agent(self, llm):
        return Agent(
            role='AI Legal Research Analyst',
            goal='Search databases of past judgments to find cases (precedents) that are similar to the current legal issue.',
            backstory=(
                "You are an AI legal researcher with expertise in Indian case law. You can quickly "
                "sift through decades of judgments from the Supreme Court and various High Courts "
                "to find the most relevant precedents for a given legal topic."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[legal_precedent_search_tool]
        )

    def make_precedent_research_task(self, agent, context):
        return Task(
            description=(
                "Based on the case summary and the identified constitutional articles, formulate a search query "
                "and use the 'Legal Precedent Search Tool' to find relevant past judgments. "
                "Summarize the findings from the search."
            ),
            expected_output=(
                "A concise summary listing the top 3-5 most relevant past court cases, "
                "including their names and a brief note on their relevance to the current case."
            ),
            agent=agent,
            context=context # Receives input from the previous agent
        )