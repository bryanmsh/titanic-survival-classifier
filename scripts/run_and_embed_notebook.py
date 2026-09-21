import json
import io
import sys
import base64
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(notebook_path):
    root_dir = os.path.abspath('.')
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    def notebook_display(*args):
        for arg in args:
            print(arg)

    global_scope = {'display': notebook_display}
    exec_count = 1

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_code = "".join(cell['source'])
            print(f"Executing cell {exec_count}...")

            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            outputs = []
            plt.close('all')

            try:
                exec(source_code, global_scope)
            except Exception as e:
                print(f"Error executing cell: {e}", file=sys.stderr)
                raise
            finally:
                sys.stdout = old_stdout

            # Capture stream stdout
            stdout_text = redirected_output.getvalue()
            if stdout_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in stdout_text.splitlines()]
                })

            # Capture any matplotlib figures created in this cell
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                outputs.append({
                    "data": {
                        "image/png": img_base64,
                        "text/plain": [f"<Figure size {fig.get_size_inches()[0]*fig.dpi}x{fig.get_size_inches()[1]*fig.dpi} with {len(fig.axes)} Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close('all')

            cell['execution_count'] = exec_count
            cell['outputs'] = outputs
            exec_count += 1

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook {notebook_path} successfully executed and updated with all outputs & plots!")

if __name__ == '__main__':
    run_notebook("titanic_pipeline.ipynb")
