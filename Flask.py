from flask import Flask, request, jsonify
import json
import commands

result  = commands.getoutput('python   sample.py')

app = Flask(__name__)


@app.route('/img/recog', methods=['GET', 'POST'])
def fout():
    if len(result) > 0:
        response = dict()
        response["msg"] = "Succ"
        response["status"] = "1"
        response["result"] = str(result)
        return json.dumps(response, ensure_ascii=False)
    else:
        response = dict()
        response["status"] = "0"
        response["msg"] = "Fail"
        return jsonify(response)
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)