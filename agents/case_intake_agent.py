# agents/case_intake_agent.py

from crewai import Agent, Task

class CaseIntakeAgents:
    # This agent no longer needs a 'tools' parameter
    def make_intake_agent(self, llm):
        return Agent(
            role='AI Paralegal for Document Review and Summarization',
            goal='Process provided legal text to produce a concise, structured summary.',
            backstory=(
                "You are a highly efficient AI paralegal, an expert in reading dense legal text "
                "to extract critical information. You create summaries that allow legal professionals "
                "to grasp the essence of a case in minutes."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    # This task now gets its data directly in the description, not through context/tools
    def make_summarization_task(self, agent, case_text):
        return Task(
            description=f"Read and summarize the following case document text:\n\n---\n\n{case_text}\n\n---",
            expected_output=(
                "A clean, well-structured summary in Markdown format, outlining the primary parties, "
                "timeline of events, main legal claims, and the relief sought."
            ),
            agent=agent
        )