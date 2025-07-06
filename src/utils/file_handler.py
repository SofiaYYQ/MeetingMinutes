import csv
from typing import List
import yaml
import openpyxl

class FileHandler:
    @staticmethod
    def read_from_txt_as_one_string(file_path:str):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
        
    @staticmethod
    def read_from_txt(file_path:str):
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file]

    @staticmethod
    def write_to_txt(file_path:str, lines:list[str], mode="w"):
        with open(file_path, mode, encoding="utf-8") as file:
            lines_with_newlines = [line.strip() + "\n" if isinstance(line, str) else str(line) for line in lines]
            file.writelines(lines_with_newlines)

    @staticmethod
    def write_to_csv(file_path:str, header:List[str], main_content:List[str], aditional_content:List[str]):
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_ALL)
            writer.writerow(header)  # Write header
            writer.writerows(main_content)
            writer.writerows(aditional_content)



    def write_to_xlsx(file_path: str, header: List[str], main_content: List[List[str]], aditional_content: List[List[str]]):
        # Crear un libro de Excel y seleccionar la hoja activa
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Escribir la cabecera
        sheet.append(header)

        # Escribir el contenido principal
        for row in main_content:
            sheet.append(row)

        # Escribir el contenido adicional
        for row in aditional_content:
            sheet.append(row)

        # Guardar el archivo
        workbook.save(file_path)

    @staticmethod
    def read_yaml(file_path: str) -> dict:
        with open(file_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
        