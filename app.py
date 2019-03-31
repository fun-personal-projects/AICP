from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from context import main
import en_core_web_sm
from abstrasumm import preprocess,extractive_summary,return_context,trends


app = Flask(__name__)
@app.route('/context', methods=['POST'])
def form_example():
	searchresult = return_context(request.json['data'])
	print(searchresult)
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
