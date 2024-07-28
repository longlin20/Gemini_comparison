import pandas as pd


def clean_code(code):
    # Remove Markdown code fences and any non-printable characters
    code = code.replace('```python', '').replace('```', '').strip()
    cleaned_code = ''.join(c if c.isprintable() or c in '\n\t ' else ' ' for c in code)
    return cleaned_code

def evaluate_code(candidate_code, test_code, entry_point):
    global_vars = {}
    candidate_code = clean_code(candidate_code)
    print(f"Generated code (repr):\n{repr(candidate_code)}\n")

    try:
        exec(candidate_code, global_vars)
    except SyntaxError as e:
        print(f"SyntaxError in generated code: {e}")
        print(f"Code with potential syntax error:\n{repr(candidate_code)}")
        return False
    except Exception as e:
        print(f"Error executing generated code: {e}")
        print(f"Code with potential execution error:\n{repr(candidate_code)}")
        return False

    candidate_func = global_vars.get(entry_point)
    if not candidate_func:
        print(f"Function {entry_point} not found in generated code.")
        return False

    test_code = clean_code(test_code)
    test_cases = [tc.strip() for tc in test_code.split('\n') if tc.strip().startswith('assert')]

    for test_case in test_cases:
        test_func_code = f"from typing import List\ndef test_func(candidate):\n    {test_case.strip()}"
        local_vars = {'candidate': candidate_func}
        print(f"Executing test case (repr):\n{repr(test_func_code)}\n")
        try:
            exec(test_func_code, {}, local_vars)
            local_vars['test_func'](candidate_func)
        except AssertionError:
            print(f"AssertionError in test case: {test_case}")
            return False
        except SyntaxError as e:
            print(f"SyntaxError in test case: {e}")
            print(f"Test case with potential syntax error:\n{test_func_code}")
            return False
        except Exception as e:
            print(f"Error executing test case '{test_case}': {e}")
            return False

    return True

def test():
    df = pd.read_excel('results/humaneval2/result_openai_humaneval_without_rag.xlsx')

    for i in range(len(df)):
        row = df.iloc[i]
        test_code = row['test code']
        entry_point = row['entry point']
        generated_code = row['answer']
        print(f"Evaluando ejemplo {i + 1}/{len(df)}")
        result = evaluate_code(generated_code, test_code, entry_point)
        print(f"result: {result}")
        df.loc[i, "results"] = result

    # Guardar los resultados en un archivo Excel
    df.to_excel('results/humaneval2/result_openai_humaneval_without_rag2.xlsx', index=False)

#test()