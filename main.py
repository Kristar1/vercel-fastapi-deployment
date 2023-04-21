from time import time
from fastapi import FastAPI, __version__
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
os.environ["OPENAI_API_KEY"] = "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"

import os
os.environ["SERPAPI_API_KEY"] = "d39e078d2268044373520d03e47d5fff46ea6ed977802d5ff95a34f79fbb6046"

from langchain.llms import OpenAI
from langchain import ConversationChain,FewShotPromptTemplate 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.callbacks import get_openai_callback
from langchain.llms.loading import load_llm
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=0)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

html = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Prepo AI Backend</title>
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon" />
    </head>
    <body>
        <div class="bg-gray-200 p-4 rounded-lg shadow-lg">
            <h1>Prepo Ai Backend</h1>
        </div>
    </body>
</html>
"""



# recognizer = sr.Recognizer()


# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)



# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)



# Preposition Examples

# First, create the list of few shot examples.
examples = [
    {"sentence": "This shop doesn’t have the books we were looking Dash", "preposition": "for"},
    {"sentence": "The governess distributed chocolates Dash the two brothers", "preposition": "between"},
    {"sentence": "Simone did not score good grades Dash his final semeste", "preposition": "in"},
    {"sentence": "Dogs are always loyal Dash their master", "preposition": "to"},
    {"sentence": "My birthday is Dash 20th January", "preposition": "on"},
    {"sentence": "The girl working Dash the departmental store is pretty", "preposition": "at"},
    {"sentence": "Netaji Subhas Chandra Bose was born Dash 23rd Januaryn", "preposition": "on"},
    {"sentence": "My little boy is not fond Dash milk", "preposition": "of"},
    {"sentence": "The cat jumped Dash the table to the sofa", "preposition": "from"},
    {"sentence": "Humpty Dumpty sat Dash a wall", "preposition": "on"},
    {"sentence": "The IAS officer was posted Dash the rural office", "preposition": "at"},
    {"sentence": "The Sun will not rise Dash 8 o’clock today", "preposition": "before"},
    {"sentence": "Mary has known Penelope Dash she was a baby girl", "preposition": "since"},
    {"sentence": "Priya’s house is Dash to Rahul’s", "preposition": "next"},
    {"sentence": "The people sitting Dash us ordered a pineapple pizza", "preposition": "opposite"},
    {"sentence": "Is the new teacher aware Dash her duties?", "preposition": "of"},
    {"sentence": "Kalpana Chawla went Dash space", "preposition": "to"},
    {"sentence": "I was stunned Dash her graceful performance", "preposition": "at"},
    {"sentence": "There’s a crack Dash the walls of the room", "preposition": "between"},
    {"sentence": "I will wait Dash the bus stop", "preposition": "at"},
    {"sentence": "you will not be able to climb dash it", "preposition": "upon"},
    {"sentence": "I'm really interested dash learning dash a new language dash the future'?", "preposition": "in, a, in"},
    {"sentence": "He was accused dash stealing dash the money dash the safe", "preposition": "of, the, from"},
    {"sentence": "She always dreamed dash living dash the countryside dash her retirement", "preposition": "of, in, for"},
    {"sentence": "I'm sorry, but I don't approve dash your plan dash quitting dash your job", "preposition": "of, of, at"},
    {"sentence": "I can't believe you're still angry dash me dash something that happened dash years ago", "preposition": "at, about, two"},
    {"sentence": "He's really good dash playing dash the piano dash ear", "preposition": "at, the, by"},
    {"sentence": "I'm really impressed dash your progress dash this project dash the last few weeks", "preposition": "by, on, over"},
    {"sentence": "The students were divided dash two groups dash the basis dash their academic performance", "preposition": "into, on, of"},
    {"sentence": "She's not used dash being dash the center dash attention", "preposition": "to, in, of"},
    {"sentence": "We're all looking forward dash spending dash a few days dash the beach", "preposition": "to, time, at"},
    {"sentence": "He was always very secretive dash his past, and never talked dash it dash anyone", "preposition": "about, with, to"},
    {"sentence": "She apologized Dash being Dash a bad mood Dash the meeting'?", "preposition": "for, in"},
    {"sentence": "I'm not sure if I'm ready dash getting dash married dash him yet", "preposition": "for, to"},
    {"sentence": "The book was written dash a perspective dash someone who had lived dash the country for many years", "preposition": "from, of, in"},
    {"sentence": "He's really good dash playing dash the guitar dash his own unique style", "preposition": "at, the, in"},
    {"sentence": "She's always been afraid dash speaking dash public, but she wants to get better dash it", "preposition": "of, in, at"},
    {"sentence": "The company was fined millions dash dollars dash violating safety regulations dash the factory", "preposition": "of, for, at"},
    {"sentence": "I'm really looking forward dash meeting dash my friends dash the airport", "preposition": "to, up, at"},
    {"sentence": "The hotel was located dash the heart dash the city, so everything was within walking distance", "preposition": "in, of, from"},
    {"sentence": "I'm sorry, but I can't agree dash you dash that point", "preposition": "with, on"},
    {"sentence": "He always insists dash doing everything dash his own, and doesn't like asking dash help", "preposition": "on, by, for"},
    {"sentence": "She was so happy dash receiving dash the award dash all her hard work", "preposition": "about, the, for"},
   
]
example_formatter_template = """
Sentence: {sentence}
Preposition: {preposition}\n
"""

example_prompt = PromptTemplate(
    input_variables=["sentence", "preposition"],
    template=example_formatter_template,
)
few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="You are a Grammer Expert. Your task now is to find the correct preposition for the given sentence.",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Sentence: {sentence}\nPreposition:",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["sentence"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n\n",
)

# def speech_to_text():
#     required=0
#     for index, name in enumerate(sr.Microphone.list_microphone_names()):
#         if "pulse" in name:
#             required= index
#     r = sr.Recognizer()
#     with sr.Microphone(device_index=required) as source:
#         r.adjust_for_ambient_noise(source)
#         print("Say something!")
#         audio = r.listen(source, phrase_time_limit=8)
#     try:
#         input = r.recognize_google(audio)
#         # with get_openai_callback() as cb:
#         #     response = agent.run(input)
#         # print("You said: " + input)
#         # print("Response: " + response)
#         # return str(response)
#         print("You said: " + input)
#         # print("Prompt: " +few_shot_prompt.format(sentence=input))
#         print("Your Answer: " + llm(few_shot_prompt.format(sentence=input)))
    
#     except sr.UnknownValueError:
#         print("Google Speech Recognition could not understand audio")
#     except sr.RequestError as e:
#         print("Could not request results from Google Speech Recognition service; {0}".format(e))



# speech_to_text()


#  API 


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://prepoai.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return HTMLResponse(html)


class PromptModel(BaseModel):
    prompt: str



@app.post("/get-preposition")
async def generate_response(prompt:PromptModel):
     print(prompt.prompt)
     print("You said: " + prompt.prompt)
     print("Your Answer: " + llm(few_shot_prompt.format(sentence=prompt.prompt)))
     return llm(few_shot_prompt.format(sentence=prompt.prompt))





@app.get('/ping')
async def hello():
    return {'res': 'pong', 'version': __version__, "time": time()}