from flask import Flask, request, jsonify
from preprocessingData import PreprocessingData
app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # Get data from POST request JSON
        data = request.json
        roomNum = data['roomNum']
        filePath = data['filePath']
        # Call your PreprocessingData function
        PreprocessingData(roomNum, filePath)

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
