import pandas as pd
import numpy as np
params = pd.read_csv('new_params.csv')

parser_add = ""
hp_add = ""
karg = ""

for i in np.arange(params.shape[0]):
	parser_add +=\
	f"""
	
parser.add_argument(
    "--{params.loc[i]['param']}", 
    type = {params.loc[i]['type']},
    help="{params.loc[i]['help']}",
    default={params.loc[i]['default']})
	
"""

	hp_add +=\
	f"""{params.loc[i]['param'].upper()} = hparams['{params.loc[i]['param']}']
"""

	karg += f"""{params.loc[i]['param']} = {params.loc[i]['param'].upper()}
"""
print(parser_add)
print("""


""")

print(hp_add)

print("""


""")


print(karg)
