from flask import Flask, request, jsonify
import argparse
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

parser = argparse.ArgumentParser(
    description="Process chatbot variables. for help run python bot.py -h"
)

parser.add_argument(
    "-s",
    "--steps",
    type=int,
    default=7,
    help="Number of steps to run the Dialogue System for",
)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('output-medium').cuda()

app = Flask(__name__)
@app.after_request
def allow_cors_if_dev(res):
    
    res.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:5500"
    res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    res.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept"
    return res

@app.route("/bot", methods=["POST"])
def bot():
    #print(request.json)
    for step in range(args.steps):
    #step=0
        incoming_msg = request.json["message"].lower()
        print(incoming_msg)
        if incoming_msg=="thank you":
            resp="I'm glad to have been of some assistance to you"
        else:
            

            new_user_input_ids = tokenizer.encode(
                incoming_msg + tokenizer.eos_token, return_tensors="pt"
            )
            bot_input_ids = (
                torch.cat([chat_history_ids, new_user_input_ids], dim=-1)\
                if step != 0\
                else new_user_input_ids
            )
            
            chat_history_ids = model.generate(
                bot_input_ids.cuda(), max_length=200,
                pad_token_id=tokenizer.eos_token_id,  
                no_repeat_ngram_size=3,       
                do_sample=True, 
                top_k=100, 
                top_p=0.7,
                temperature = 0.8
            ).cpu()

            resp=str(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
        print(resp)
        return jsonify({"resp":resp})



if __name__ == "__main__":
    app.run()