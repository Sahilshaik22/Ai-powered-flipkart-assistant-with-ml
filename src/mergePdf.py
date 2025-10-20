from PyPDF2 import PdfMerger
import os
merger = PdfMerger()
path_dir = ".\Data"
output_path = os.path.join(path_dir,"Flipkart_policies.pdf")

pdf_list = [f for f in os.listdir(path_dir) if f.lower().endswith(".pdf")]

for file_name in pdf_list:
    file_path = os.path.join(path_dir,file_name)
    merger.append(file_path)
    
merger.write(output_path)
merger.close()
print("merging is finished")

for file_name in pdf_list:
    file_path = os.path.join(path_dir,file_name)
    try:
        if os.path.exists(file_path) and file_path != output_path:
            os.remove(file_path)
            print(f"Deleted old file {file_name}")
    except Exception as e:
        print(f"cloud not deleted file {file_name}:{e}")
print("cleanup completd check folder")