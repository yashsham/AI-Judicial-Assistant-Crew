# agents/synthesis_agent.py

from crewai import Agent, Task

class SynthesisAgents:
    def make_synthesis_agent(self, llm):
        return Agent(
            role='Senior AI Judicial Clerk',
            goal='Combine the case summary, constitutional analysis, and legal precedents into a single, comprehensive Preliminary Judicial Note.',
            backstory=(
                "You are a senior AI judicial clerk with a talent for synthesis. You take "
                "disparate pieces of information—case facts, constitutional law, and historical "
                "judgments—and weave them into a clear and logical report that provides a "
                "360-degree view of a legal matter."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def make_synthesis_task(self, agent, context):
        return Task(
            description=(
                "Synthesize all the provided information into a final, well-structured 'Preliminary Judicial Note' in Markdown format. The note must include:\n"
                "1. A section for the 'Case Summary' which is final answer of Agent AI Paralegal for Document Review and Summarization.\n"
                "2. A section for 'Relevant Constitutional Articles' that lists the articles found.\n"
                "3. A section for 'Relevant Legal Precedents' that summarizes the past cases.\n"
                "4. A final 'Conclusion' section that briefly ties all the points together."
            ),
            expected_output=(
                "A complete and polished Preliminary Judicial Note in Markdown format, "
                "containing all the required sections and synthesized information."
            ),
            agent=agent,
            context=context # Receives input from all previous agents
        )