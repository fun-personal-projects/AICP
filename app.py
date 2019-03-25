from flask import Flask
from flask import request
from flask import jsonify
from context import main
import en_core_web_sm
from abstrasumm import preprocess,extractive_summary


app = Flask(__name__)
@app.route('/context', methods=['POST'])
def form_example():
        searchresult = main(request.json['data'])
        print(searchresult)
   	return jsonify({"data":searchresult})
@app.route('/summary', methods=['POST'])
def summary():
        nlp = en_core_web_sm.load()
        docu = preprocess(request.json['data'])
        summ = extractive_summary(docu)
        return jsonify({"summary":summ})

        

if __name__ == '__main__':
    app.run(host='0.0.0.0')
