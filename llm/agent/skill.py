import os
import logging

logging.basicConfig(
    filename="info.log",
    level=logging.INFO,
    format="%(message)s",
    force=True
)
log = logging.getLogger()

user_input = input()
skill_prompt = open("skills/skill.md").read()
class LLM:
    def generate(self, prompt):
        print(prompt)
        return prompt
llm = LLM()

informat = llm.generate(skill_prompt + "\n\n" + user_input)
log.info("Skill executed successfully." + "\n\n\n" + informat)
