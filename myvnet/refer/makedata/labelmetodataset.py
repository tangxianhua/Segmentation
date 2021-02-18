#json to dataset
import os

filepath = r"C:\Users\86198\Desktop\秸秆焚烧\Nice\json"
for root, dirs, files in os.walk(filepath):
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            json_path = os.path.join(root, file)
            cmd1 = r"labelme_json_to_dataset {}".format(json_path)
            os.system(cmd1)


