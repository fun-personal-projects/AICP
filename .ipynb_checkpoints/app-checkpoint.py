from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from context import main
import en_core_web_sm
from abstrasumm import preprocess,extractive_summary,return_context,trends,calendar_entry


app = Flask(__name__)
@app.route('/context', methods=['POST'])
def form_example():
	searchresult = context_json(request.json['data'])
    # this now gives a json
	# print(searchresult)
    calendar_entry()

	return searchresult

@app.route('/summary', methods=['POST'])
def summary():
	nlp = en_core_web_sm.load()
	data = request.json['data']
	docu = preprocess(data)
	summ = extractive_summary(data,docu)

	return jsonify({"summary":summ})

@app.route('/wordcloud',methods=['POST'])
def findimage():
	trends(request.json['data'])
	return send_file('trends.png')

if __name__ == '__main__':
	app.run(host='0.0.0.0')
