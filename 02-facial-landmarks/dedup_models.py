from weights import weights

# ================================
# Notes
# ================================
#
# To use this file:
#
#    python dedup_models.py | xargs -L 1 -I {} rm "models/"{}
#

df = weights()

files_to_keep = df.groupby('unique_loss').first().filename.values.tolist()

files_to_delete = df.filename[~df.filename.isin(files_to_keep)]

print('\n'.join(files_to_delete))

# ================================
# Manual verification of accuracy
# ================================
#
# print(df.head())
# print(df.groupby('unique_loss').first())
#
# df.to_csv('sorted.csv')
# (df.groupby('unique_loss').first()
#  .sort_values(['model', 'val_loss', 'step'])
#  .to_csv('dedup.csv'))
