
# coding: utf-8
from nilearn import datasets
from nilearn import plotting

from nilearn import image
smoothed_img = image.load_img('/Users/finnrabe/Downloads/BrainHack/Data/subj1/bold.nii')
print("smoothed image loaded")

# In[88]:
from nilearn.plotting import plot_stat_map, show
mask_filename = image.load_img('/Users/finnrabe/Downloads/BrainHack/Data/subj1/mask4_vt.nii') #haxby_dataset.mask_vt[0]
ana = image.load_img('/Users/finnrabe/Downloads/BrainHack/Data/subj1/anat.nii')
print("mask and anatomical image loaded")
# Let's visualize it, using the subject's anatomical image as a
# background
# plotting.plot_roi(mask_filename, bg_img=ana,
#                  cmap='Paired')
# show()
# print("finish plotting")

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename)
print(masker)
# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(smoothed_img)
print(fmri_masked.shape)

import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

conditions = behavioral['labels']
condition_mask = conditions.isin(['face', 'house'])

# We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]
fmri_masked.shape

conditions = conditions[condition_mask]
print(conditions.shape)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(fmri_masked, conditions)
prediction = svc.predict(fmri_masked)
print((prediction == conditions).sum() / float(len(conditions)))


from sklearn.model_selection import KFold

cv = KFold(n_splits=5)
# The "cv" object's split method can now accept data and create a
# generator which can yield the splits.
for train, test in cv.split(X=fmri_masked):
    conditions_masked = conditions.values[train]
    svc.fit(fmri_masked[train], conditions_masked)
    prediction = svc.predict(fmri_masked[test])
    print((prediction == conditions.values[test]).sum()
           / float(len(conditions.values[test])))


from sklearn.model_selection import cross_val_score
session_label = behavioral['chunks'][condition_mask]

# By default, cross_val_score uses a 3-fold KFold. We can control this by
# passing the "cv" object, here a 5-fold:
cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)

# To leave a session out, pass it to the groups parameter of cross_val_score.
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()
cv_score = cross_val_score(svc,
                           fmri_masked,
                           conditions,
                           cv=cv,
                           groups=session_label,
                           )
print(cv_score)


# In[78]:


coef_ = svc.coef_


# In[79]:


coef_img = masker.inverse_transform(coef_)


# In[83]:


coef_img.to_filename('haxby_svc_weights2.nii.gz')


# In[84]:


from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              title="SVM weights", display_mode="yx")

show()

