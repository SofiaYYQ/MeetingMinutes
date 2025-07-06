

from typing import List
from utils.evaluation_mode_validator import EvaluationModeValidator
from utils.file_handler import FileHandler

class AccuracyEvaluator:
    @staticmethod
    def compare_lists(list1:str, list2:str):
        if len(list1) != len(list2):
            raise Exception("Lists do not have same length")
        num_elements = len(list1)
        results = []
        gemma_validator_results = []
        for idx in range(num_elements):
            correct = False
            # Comprobar que no sea una lista de elementos
            if "," in list1[idx]:
                if "," in list2[idx]:
                    parts_list1 = [part.strip() for part in list1[idx].split(",")]
                    parts_list2 = [part.strip() for part in list2[idx].split(",")]
                    correct = set(parts_list1) == set(parts_list2) 
                    gemma_validator_results.append("")
                else:
                    result = EvaluationModeValidator().compare(list1[idx], list2[idx])
                    try:
                        correct = result["respuesta"] == "Sí"
                        gemma_validator_results.append(str(result))
                    except Exception as e:
                        print(f"Error: {type(e).__name__} - {e}. Cannot retrieve formatted answer.")
                        gemma_validator_results.append("")
            else:
                gemma_validator_results.append("")
                correct = list1[idx] == list2[idx]

            if correct:
                results.append("Correct")
            else:
                results.append("Incorrect")
        
        num_matches = results.count("Correct")
        accuracy = (num_matches / num_elements) if num_elements > 0 else 0
        
        return results, num_matches, accuracy, gemma_validator_results
    
    @staticmethod
    def get_accuracy_results(file_path:str, questions:List[str], expected:List[str], real:List[str], formatted_real:List[str]):
        comparation_results = AccuracyEvaluator.compare_lists(expected, formatted_real)
        FileHandler.write_to_csv(
            file_path, 
            ["Question", "Expected Answer", "Real Answer (formatted)", "Correct/Incorrect", "Notas", "Real Answer"],
            zip(questions, expected, formatted_real, comparation_results[0], comparation_results[3], real),
            [("", "", "Total correct", comparation_results[1]),("", "", "Accuracy", comparation_results[2])]
        )
        FileHandler.write_to_xlsx(
            file_path.replace(".csv", ".xlsx"), 
            ["Question", "Expected Answer", "Real Answer (formatted)", "Correct/Incorrect", "Notas", "Real Answer"],
            zip(questions, expected, formatted_real, comparation_results[0], comparation_results[3], real),
            [("", "", "Total correct", comparation_results[1]),("", "", "Accuracy", comparation_results[2])]
        )

    @staticmethod
    def get_results(file_path:str, questions:List[str], expected:List[str], real:List[str]):
        FileHandler.write_to_csv(
            file_path, 
            ["Question", "Expected Answer", "Real Answer"],
            zip(questions, expected, real),
            []
        )
        FileHandler.write_to_xlsx(
            file_path.replace(".csv", ".xlsx"), 
            ["Question", "Expected Answer", "Real Answer"],
            zip(questions, expected, real),
            []
        )