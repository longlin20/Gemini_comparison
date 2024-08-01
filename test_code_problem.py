import pandas as pd


def clean_code(code):
    """
    Remove Markdown code fences and non-printable characters from the code.

    Args:
        code (str): The code to be cleaned.

    Returns:
        str: The cleaned code.
    """
    code = code.replace('```python', '').replace('```', '').strip()
    cleaned_code = ''.join(c if c.isprintable() or c in '\n\t ' else ' ' for c in code)
    return cleaned_code


def evaluate_code(code, test_code, entry_point):
    """
    Evaluate the candidate code against the test code.

    Args:
        code (str): The candidate code.
        test_code (str): The test code.
        entry_point (str): The entry point of the candidate code.

    Returns:
        bool: True if the code passes all test cases, False otherwise.
    """
    global_vars = {}
    code = clean_code(code)

    try:
        exec(code, global_vars)
    except SyntaxError as e:
        print(f"SyntaxError in generated code: {e}")
        print(f"Code with potential syntax error:\n{repr(code)}")
        return False
    except Exception as e:
        print(f"Error executing generated code: {e}")
        print(f"Code with potential execution error:\n{repr(code)}")
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

def test_evaluation(filename):
    """
    Test the evaluation of the code.
    """
    df = pd.read_excel(filename)

    for i, row in enumerate(df.iterrows()):
        test_code, generated_code, entry_point = row[1]['test code'], row[1]['answer'], row[1]['entry point']
        print(f"Evaluating example {i + 1}/{len(df)}")
        result = evaluate_code(generated_code, test_code, entry_point)
        print(f"Result: {result}")
        df.loc[i, "results"] = result

    df.to_excel(f'{filename[:-5]}_2.xlsx', index=False)