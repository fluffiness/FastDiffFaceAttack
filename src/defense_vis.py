import matplotlib.pyplot as plt
import json
import numpy as np

defended_asrs_file = "../logs/07-27_03:11:15_robustness_test/defended_asrs.json"
with open(defended_asrs_file, 'r') as f:
    defended_asrs = json.load(f)

attack_methods = ["pgd", "amt-gan", "diffprot", "ours"]
attack_method_names = ["PGD", "AMT-GAN", "DiffProtect", "Ours"]
test_model_names = ['mobile_face', 'ir152', 'irse50', 'facenet']
test_model_names_full = ["MobileFace", "IR152", "IRSE50", "FaceNet"]
defense_methods = ['undefended', 'feature_squeezing', 'gaussian_blur', 'median_blur', 'jpeg']
defense_method_names = ["none", "feature\nsqueezing", 'gaussian\nblur', 'median\nblur', 'jpeg']

ylims = [(0, 80), (0, 50), (0, 80), (0, 30)]
logend_locs = ['lower left', 'lower left', 'lower left', 'upper right']

defended_asr_data_seq = {tmn: {am: [defended_asrs[am][tmn][dm] for dm in defense_methods] for am in attack_methods} for tmn in test_model_names}
# print(defended_asr_data_seq)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.3)
for i_tmn, tmn in enumerate(test_model_names):
    ax = axs[i_tmn//2, i_tmn%2]
    ax.set(title=test_model_names_full[i_tmn])
    for i_am, am in enumerate(attack_methods):
        asr_data = [defended_asrs[am][tmn][dm] for dm in defense_methods]
        ax.plot(defense_method_names, asr_data, marker='o', label=attack_method_names[i_am])
    ax.set_ylim(*ylims[i_tmn])
    ax.legend(loc=logend_locs[i_tmn], ncol=2)

fig.savefig("defense_plots.png", bbox_inches='tight')