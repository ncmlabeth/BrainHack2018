from sklearn.model_selection import LeaveOneOut
from nilearn import plotting, datasets
from nilearn.input_data import NiftiMasker
import numpy as np
from sklearn.svm import SVC
# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filenames: one for each subject
fmri_filename = haxby_dataset.func[0]

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      fmri_filename)  # 4D data

###########################################################################
# Convert the fMRI volume's to a data matrix
# ..........................................
#
# We will use the :class:`nilearn.input_data.NiftiMasker` to extract the
# fMRI data on a mask and convert it to data series.
#
# The mask is a mask of the Ventral Temporal streaming coming from the
# Haxby study:
mask_filename = haxby_dataset.mask_vt[0]

# Let's visualize it, using the subject's anatomical image as a
# background

plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                 cmap='Paired')

###########################################################################
# Now we use the NiftiMasker.
#
# We first create a masker, giving it the options that we care
# about. Here we use standardizing of the data, as it is often important
# for decoding
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)



import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
print(behavioral)

###########################################################################
# Retrieve the experimental conditions, that we are going to use as
# prediction targets in the decoding
conditions = behavioral['labels']
print(conditions)


session_label = behavioral['chunks']


# create a new dataframe with only the chunk 0 and 
# condition with associated rest
acc_lst = []
for run in range(12):
    chunk0 = behavioral[behavioral["chunks"] == run]

    chunk_rest = behavioral[behavioral["labels"] == "rest"]
    chunk0_rest = chunk_rest[chunk_rest["chunks"] == run]

    chunk_cond = behavioral[behavioral["labels"] != "rest"]
    chunk0_cond = chunk_cond[chunk_cond["chunks"] == run]

    labels_chunk = np.unique(chunk0_cond["labels"], return_index=True)
    sorted_index = np.sort(labels_chunk[1])
    labels_chunk = chunk0_cond["labels"][chunk0_cond["labels"].index[sorted_index]]

    # extract different rest series
    list_ind_all_rest = []
    list_ind = []
    for ind, i in enumerate(chunk0_rest.index):
        if ind > 0:
            prev_ind = chunk0_rest.index[ind - 1]
            if i == prev_ind + 1:
                list_ind.append(i)

            else:
                list_ind_all_rest.append(list_ind)
                list_ind = []
                list_ind.append(i)
        else:
            list_ind.append(i)

    # extract different conditions series
    grouped = chunk0_cond.groupby(chunk0_cond["labels"])
    list_ind_all_cond = []
    for i in labels_chunk:
        list_ind_all_cond.append(grouped.get_group(i).index)

    grouped_index = []
    for ind, i in enumerate(labels_chunk):
        index_rest = list_ind_all_rest[ind]
        index_cond = list_ind_all_cond[ind]
        concat = np.concatenate([index_rest, index_cond])
        grouped_index.append(concat)

    loo = LeaveOneOut()
    svc = SVC(kernel='linear')

    # extract data from index
    for g in range(0, len(grouped_index)):
        print ("GROUP {}".format(g))
        group = behavioral.loc[grouped_index[g]]
        condition = group["labels"]
        fmri_masked_group = fmri_masked[grouped_index[g]]
        X = fmri_masked_group
        y = condition
        y = y.reset_index(drop=True)

        # deal with class imbalance
        ind_0 = np.where(condition == "rest")[0]
        ind_1 = np.where(condition != "rest")[0]
        n_1 = len(ind_1)
        ind_0_upsampled = np.random.choice(ind_0, size=n_1, replace=True)
        # Join together the vectors of target
        y = np.hstack((y[ind_0_upsampled], y[ind_1]))
        # Join together the arrays of features
        X = np.vstack((X[ind_0_upsampled,:], X[ind_1,:]))

        # leave one out CV
        for train, test in loo.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            svc.fit(X_train, y_train)
            pred = svc.predict(X_test)
            acc = ((pred == y_test).sum()/ float(len(y_test)))
            acc_lst.append(acc)

    # group = behavioral.loc[grouped_index[0]]
    # condition = group["labels"]
    # fmri_masked_group = fmri_masked[grouped_index[0]]
    # X = fmri_masked_group
    # y = condition
    #
    # for train, test in loo.split(X):
    #     X_train, X_test = X[train], X[test]
    #     y_train, y_test = y[train], y[test]
    #     svc.fit(X_train, y_train)
    #     pred = svc.predict(X_test)
    #     acc2 = ((pred == y_test).sum()/ float(len(y_test)))
    #     acc_lst.append(acc2)

print('mean accurarcy:', np.mean(acc_lst))

############################
# plot confusion matrices #
###########################

np.random.seed(0)

from sklearn.metrics import confusion_matrix

class_names = np.unique(conditions, return_counts=True)[0]
y_pred = predictions
y_true = conditions
        
cm = confusion_matrix(y_true, y_pred)

cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage

plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
            horizontalalignment="center",
            color= "black", fontsize=8)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=12, rotation=90)
plt.yticks(tick_marks, class_names, fontsize=12)
plt.show()
