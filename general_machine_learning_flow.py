import numpy as np
import matplotlib, pickle
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from tensorflow import keras
import seaborn as sns
np.set_printoptions(suppress=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import umap.umap_ as umap  # Make sure umap-learn is installed: pip install umap-learn
from sklearn.preprocessing import RobustScaler, LabelEncoder

from tensorflow.keras import layers, regularizers, callbacks, Model, Input
import tensorflow as tf


every_spectrum_wl_axis  = """ INPUT WAVELENGTH GRID """ # consistent to all objects
every_spectrum_flux     = """ INPUT NORMALISED FLUX FROM ALL SPECTRA WITH A COMMON WAVELENGTH GRID """  # must be a consistent normalisation procedure to all spectra  
every_spectrum_classify = """ INPUT CLASSIFICATIONS """ # strings


# Step 1: Standardize blocks
X_combined = RobustScaler().fit_transform(every_spectrum_flux)
sns.set(style="whitegrid")

le = LabelEncoder()
y = le.fit_transform(every_spectrum_classify)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_combined, y, indices, test_size=0.2, random_state=42)  #test_size = 1 means that 0% is being used as training data

n_samples = X_combined.shape[0]
indices = np.arange(n_samples)



specname_ylabel = {}
for val in np.unique(every_spectrum_classify):
	arg = np.argwhere(every_spectrum_classify==val)[0][0]
	specname_ylabel[str(y[arg])] = str(every_spectrum_classify[arg])

specname_ylabel_reversed = {}
for cnt, val in enumerate(specname_ylabel):
	specname_ylabel_reversed[specname_ylabel[str(cnt)]] = val



test_performance=True
plot_cm_other=True
plot_umap=True


"""
=============================================================================================================
CODE TO DO MACHINE LEARNING
============================================================================================================= 
"""
probability_threshold = 0.7

if test_performance:
	specname_ylabel[str(len(np.unique(every_spectrum_classify))+1)] = "P<"+str(probability_threshold)

	
	atmos_map = {"CV":0, "DA":1, "DAH":2, "DB":3, "DC":4, "DQ":5, "DZ":6, "DO/DAO":7}
	atmos_map_reversed = {0:"CV", 1:"DA", 2:"DAH", 3:"DB", 4:"DC", 5:"DQ", 6:"DZ", 7:"DO/DAO"}

	def prepare_labels(y_raw):
		y_atmos, y_Z = [], []
		
		for lbl in y_raw:
			lbl = specname_ylabel[str(lbl)]
			# -------- Atmosphere --------
			if lbl=="DA" or lbl=="DAZ/DZA":                   atmos = atmos_map["DA"]
			elif lbl=="DB" or lbl=="DBZ/DZB":                 atmos = atmos_map["DB"]
			else:                                             atmos = atmos_map[lbl]  # works for DO, DQ, DC, DZ, CV
			y_atmos.append(atmos)
			
			# -------- Metals --------
			if lbl in ["DAZ/DZA", "DBZ/DZB", "DZ"]:           zflag = 1   # has metals
			elif lbl == "CV":                                 zflag = -1  # ignore CVs in metals head
			else:                                             zflag = 0   # no metals
			y_Z.append(zflag)
			
		return np.array(y_atmos), np.array(y_Z)

	y_atmos, y_Z = prepare_labels(y_train)
	
	
	

	inputs = Input(shape=(X_train.shape[1],))
	spec_input = layers.Input(shape=(X_train.shape[1],), name="spectrum")
	
	# ---- CNN branch ----
	x = layers.Reshape((X_train.shape[1], 1))(spec_input)
	    
	# Multi-scale convolution to capture narrow + broad lines
	k1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)   # very narrow features
	k2 = layers.Conv1D(32, kernel_size=11, padding='same', activation='relu')(x)  # medium features
	k3 = layers.Conv1D(32, kernel_size=23, padding='same', activation='relu')(x)  # broad features

	x = layers.Concatenate()([k1, k2, k3])
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	# Additional conv layers to refine feature extraction
	x = layers.Conv1D(64, kernel_size=15, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	x = layers.Conv1D(128, kernel_size=23, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.GlobalMaxPooling1D()(x)

	# ---- Fully connected classifier ----
	x = layers.Dense(256, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.4)(x)

	x = layers.Dense(128, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.3)(x)

	x = layers.Dense(64, activation='relu')(x)
	x = layers.BatchNormalization()(x)

	# Atmosphere head
	out_atmos = layers.Dense(len(atmos_map), activation="softmax", name="atmosphere")(x)

	# Metals head (yes/no)
	out_Z = layers.Dense(2, activation="softmax", name="metals")(x)
	
	model = Model(inputs={"spectrum": spec_input}, outputs={"atmosphere": out_atmos, "metals": out_Z})

	# Masked loss for metals (ignore -1 labels)
	def masked_sparse_ce(y_true, y_pred):
		"""
		Sparse categorical crossentropy that ignores samples with label == -1.
		y_true: (batch,) integers 0 or 1, or -1 to ignore
		y_pred: (batch, num_classes) softmax probabilities
		"""
		y_true = tf.cast(y_true, tf.int32)

		# Create mask: 1 for valid labels, 0 for -1
		mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
		y_true_safe = tf.where(y_true < 0, 0, y_true)

		# Compute sparse categorical crossentropy per sample
		loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred)
		loss = loss * mask

		# Normalize by number of valid samples
		return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-6)

	model.compile(
	    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
	    loss={"atmosphere": "sparse_categorical_crossentropy",
		  "metals": masked_sparse_ce},
	    metrics={"atmosphere": "accuracy", "metals": "accuracy"}
	)

	
	
	from sklearn.utils.class_weight import compute_class_weight
	
	# weights = 1 for valid samples, 0 for ignored (-1)
	metals_weights = np.where(y_Z == -1, 0.0, 1.0)

	
	weight=np.array([])
	for bbb, atmo, zz in zip(y_train, y_atmos, y_Z):
		if bbb==1 and zz!=1:
			weight=np.append(weight, 0.25)
		elif (bbb==1 and zz==1) or bbb==0:
		    weight=np.append(weight, 3)
		else:
			weight=np.append(weight,1)
	
	
	sample_weight = {
	    "atmosphere": weight,
	    "metals": np.where(y_Z == -1, 0.0, 1.0)   # 0 = ignore, 1 = include
	}
	
	
	
	es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
	rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
	mc = callbacks.ModelCheckpoint("best_model_advanced.h5", monitor="val_loss", save_best_only=True)
	

	history = model.fit(
	    X_train,
	    {"atmosphere": y_atmos, "metals": y_Z},
	    sample_weight=sample_weight,
	    validation_split=0.1, epochs=100, batch_size=64,
	    callbacks=[es, rlrop, mc], shuffle=True, verbose=1
	)
	
	
	
	# Predictions
	model = keras.models.load_model("best_model_advanced.h5")
	preds = model.predict(X_test)
	
	y_preds_atmos = preds["atmosphere"]
	y_preds_Z = preds["metals"]
	
	y_pred_atmos = y_preds_atmos.argmax(axis=1)
	y_pred_Z = y_preds_Z.argmax(axis=1)
	
	
	
	
	
	
	
	spectype_pred_atmos, probas = [], []
	for cnt_pred, (i, j) in enumerate(zip(y_pred_atmos, y_pred_Z)):
		if atmos_map_reversed[i]=="CV" or atmos_map_reversed[i]=="DAB/DBA" or atmos_map_reversed[i]=="DC" or atmos_map_reversed[i]=="DO" or atmos_map_reversed[i]=="DQ" or atmos_map_reversed[i]=="DAH" or atmos_map_reversed[i]=="DO/DAO":
			spectype_pred_atmos.append(atmos_map_reversed[i])
		elif j==1 and not atmos_map_reversed[i]=="DZ":
			toappend=atmos_map_reversed[i]+"Z"
			if toappend=="DAZ": spectype_pred_atmos.append("DAZ/DZA")
			elif toappend=="DBZ": spectype_pred_atmos.append("DBZ/DZB")
			else: raise ValueError(i, j, atmos_map_reversed[i])
		elif "DA" in atmos_map_reversed[i] or "DB" in atmos_map_reversed[i] or atmos_map_reversed[i]=="DZ":
			spectype_pred_atmos.append(atmos_map_reversed[i])
			
		else:    raise ValueError(atmos_map_reversed[i])
		
		
		CVprob=y_preds_atmos[cnt_pred][0]
		DAprob=y_preds_atmos[cnt_pred][1]*y_preds_Z[cnt_pred][0]
		DAZprob=y_preds_atmos[cnt_pred][1]*y_preds_Z[cnt_pred][1]
		DAHprob=y_preds_atmos[cnt_pred][2]
		DBprob=y_preds_atmos[cnt_pred][3]*y_preds_Z[cnt_pred][0]
		DBZprob=y_preds_atmos[cnt_pred][3]*y_preds_Z[cnt_pred][1]
		DCprob=y_preds_atmos[cnt_pred][4]
		DQprob=y_preds_atmos[cnt_pred][5]
		DZprob=y_preds_atmos[cnt_pred][6]
		DOAprob=y_preds_atmos[cnt_pred][7]
		
		
		prob=np.array([CVprob, DAprob, DAHprob, DAZprob, DBprob, DBZprob, DCprob, DQprob, DZprob, DOAprob])
		probas.append(prob)
	
	
	
	spectype_pred_atmos=np.asarray(spectype_pred_atmos)
	probas=np.asarray(probas)
	
	
	ypred_on_original_grid = np.array([], dtype=float)
	
	for aaa in spectype_pred_atmos:
		ypred_on_original_grid = np.append(ypred_on_original_grid, specname_ylabel_reversed[aaa])
	
	
	y_pred=ypred_on_original_grid.astype(int)
	
	
	with open('best_model_advanced_XtrainYtrainYpredXtestYtest.pkl', 'wb') as f:
		pickle.dump([X_train, y_train, y_pred, X_test, y_test, probas], f)
			


""" =============================================================================================================
    CODE TO PLOT CONFUSION MATRIX
    =============================================================================================================
"""
    
	
if plot_cm_other:
	with open('best_model_advanced'+'_XtrainYtrainYpredXtestYtest.pkl', 'rb') as f:
		(X_train, y_train, y_pred, X_test, y_test, probas) = pickle.load(f)
	
	
	incorrect_idx, correct_idx = np.array([], dtype=bool), np.array([], dtype=bool)
	
	for cnt, (i, ii) in enumerate(zip(y_pred, y_test)):
		if i==ii:
			incorrect_idx = np.append(incorrect_idx, False)
			correct_idx = np.append(correct_idx, True)
		else:
			incorrect_idx = np.append(incorrect_idx, True)
			correct_idx = np.append(correct_idx, False)


	index_orig_test_set=np.array([])
		
	
	
	if np.amax(probas)!=0:  #  make P<0.9 things classes as "unknown"
		unknown_class_index = len(le.classes_)+1
		max_probas = np.max(probas, axis=1)
		final_preds_int = np.where(max_probas < probability_threshold, unknown_class_index, y_pred)
		y_pred=final_preds_int
		del final_preds_int
		
		all_class_names = list(le.classes_) + ["P<"+str(probability_threshold)]
		all_labels = list(range(len(le.classes_))) + [unknown_class_index]
	
	print("Accuracy:", accuracy_score(y_test, y_pred))
	
	try: print("Classification Report:\n", classification_report(y_test, y_pred, target_names=all_class_names, labels=all_labels))
	except: None
    
	
	speclabel=[]

	for i in y_pred:
		if i==0:  speclabel.append("CV")
		elif i==1:  speclabel.append("DA")
		elif i==2:  speclabel.append("DAH")
		elif i==3:  speclabel.append("DAZ/DZA")
		elif i==4:  speclabel.append("DB")
		elif i==5:  speclabel.append("DBZ/DZB")
		elif i==6:  speclabel.append("DC")
		elif i==7:  speclabel.append("DQ")
		elif i==8:  speclabel.append("DZ")
		elif i==9:  speclabel.append("DO/DAO")
		elif i==10:  speclabel.append("P<"+str(probability_threshold))

	
	# Compute and normalize confusion matrix
	cm = confusion_matrix(y_test, y_pred, labels=all_labels)
	cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
	

	# Prepare annotations
	annot = np.empty_like(cm).astype(str)
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			count = cm[i, j]
			if all_class_names[-1].startswith("P"):
				frac = cm[i, j] / np.sum(cm[i, :-1])  #frac = cm[i, j] / np.sum(cm[i, :])
			else:
				frac = cm[i, j] / np.sum(cm[i, :])
			
			#frac = cm_normalized[i, j]
			if not j==cm.shape[1]-1 or not all_class_names[-1].startswith("P"):
				annot[i, j] = f"{count}\n({frac:.3f})" if count > 0 else ""
				cm_normalized[i, j] = frac
			else:
				annot[i, j] = f"{count}" if count > 0 else ""
				cm_normalized[i, j] = 0

	cm = cm[:-1, :]
	cm_normalized = cm_normalized[:-1, :]
	annot = annot[:-1, :]
	
	with open('confusion_matrix_info.pkl', 'wb') as f:
		pickle.dump([cm, cm_normalized, y_test, y_pred], f)  


	# Masks
	mask_diag = np.zeros_like(cm_normalized, dtype=bool)
	np.fill_diagonal(mask_diag, True, wrap=False)  # still mark diagonal True up to min dimension
	mask_offdiag = ~mask_diag
	

	fig = plt.figure(figsize=(12, 8))
	gs = GridSpec(1, 2, width_ratios=[0.95, 0.05], wspace=0.1)
	ax_bluebar = fig.add_subplot(gs[1]);   ax_matrix = fig.add_subplot(gs[0])
	
		
	colors = [
		(0.0, "#FFF4ED"),  # light cream from your image
		(0.5, "#6CA6CD"),  # medium sky blue
		(1.0, "#1D2475"),  # dark blue
	]
		
	beige_to_blue = LinearSegmentedColormap.from_list("beige_to_blue", colors)
		
	# Overlay blue heatmap
	sns.heatmap(cm_normalized, annot=annot, fmt="",  cmap=beige_to_blue, cbar=True, ax=ax_matrix,  xticklabels=all_class_names, yticklabels=all_class_names[:-1],
		    linewidths=0.5, linecolor='gray',  cbar_ax=ax_bluebar,  vmin=0, vmax=1)


	ax_matrix.set_xlabel("Predicted Label");       ax_matrix.set_ylabel("True Label")
	
	plt.tight_layout()
	plt.savefig("confusion_matrix.pdf", bbox_inches='tight', dpi=1000)
	plt.show();  plt.close()
	
	
	
	
	
"""
=============================================================================================================
CODE TO DO UMAP
============================================================================================================= 
"""
		

if plot_umap:
    # UMAP
	reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
	embedding = reducer.fit_transform(X_combined)
	
	
	""" ========= PLOT THE RESULTS ========= """

	colours=["r", 'b', "cyan", 'g', 'r', 'k', 'magenta', 'yellow', "grey", "orange", 'lightgreen', 'lightsalmon', "lightgreen"]

	import matplotlib.gridspec as gridspec
	
	fig, ax0 = plt.subplots(1)
	

	# True labels
	for cnt, i in enumerate(np.unique(speclabel)):
		mask=speclabel==i
		if i=="CV":
			ax0.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=40,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=100, rasterized=True)
			ax1.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=20,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=100, rasterized=True)
			ax2.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=20,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=100, rasterized=True)
		elif "P<" in i:
			i="P$\,$<$\,$0.7"
			ax0.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=40,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=20, rasterized=True)
			ax1.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=20,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=20, rasterized=True)
			ax2.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], marker="x", cmap='tab10', s=20,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=20, rasterized=True)
		elif i=="DA":
			ax0.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=1,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
			ax1.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=5,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
			ax2.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=5,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
		else:
			ax0.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=10,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
			ax1.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=5,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
			ax2.scatter(embedding[:, 0][mask], embedding[:, 1][mask], c=colours[cnt], cmap='tab10', s=5,label=str(i).replace("DOA/DAO", "DO/DAO"), zorder=50, rasterized=True)
	ax0.set_xlabel("UMAP Dimension 1")
	ax0.set_ylabel("UMAP Dimension 2")
	ax0.legend(loc="upper right", ncol=2, columnspacing=0.5)
	ax0.grid(False)
	
	plt.savefig("UMAPplot.pdf", bbox_inches='tight', dpi=1000)
	#plt.show()
	plt.close()
