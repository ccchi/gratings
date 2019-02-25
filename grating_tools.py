import numpy as np
from astropy.io import fits
import scipy.stats
import scipy.interpolate
import scipy.ndimage
import os.path
    
'''
Fits a line through diffraction bragg peaks. Bragg peaks are identified by identifying groups of pixels with intensities greater
than the median intensity of the entire image by some amount. The line is fit using the Theil-Sens estimation method to reduce the
effect of outliers (e.g. anomalously bright spots) on the line.

Arguments:
    img_data: 2D array containing intensity values
    sig: how many median absolute deviations away from the median intensity of the image for a pixel to be counted as a bragg peak
    percentile: the percentage of least intense bragg peaks discarded prior to line fitting
    min_peaks: the minimum number of peaks to try to find. If the number of peaks detected is less than min_peaks, the peak finding procedure will be repeated with a lower sig level until either the number of peaks detected meets or exceeds min_peaks, or min_sig is reached.
    min_sig: the minumum number of median absolute deviations before terminating the peak finding procedure. If sig < min_sig, the
    
Returns:
    line: a tuple containing values in indicies
        0: estimated slope
        1: estimated y intercept
        2: lower bound on confidence interval for estimated slope
        3: upper bound on the confidence interval for estimated slope
    If the estimated slope is undefined (i.e. the line is vertical), the tuple returned will contain
        0: inf
        1: estimated x intercept
        2: lower bound on confidence interval for estimated slope if x and y were swapped
        3: upper bound on the confidence interval for estimated slope if x and y were swapped
'''
def find_line(img_data,sig=25,percentile=60,min_peaks=10,min_sig=3):
    img=np.log(img_data)
    num_peaks=0
    while(num_peaks<min_peaks):
        mask=np.abs(img-np.median(img))>sig*np.median(np.abs(img-np.median(img)))
        label_img,num_label=scipy.ndimage.label(mask)
        intensity=scipy.ndimage.maximum(img_data,label_img,range(num_label+1))
        intensity_mask=intensity<np.percentile(intensity,percentile)
        remove=intensity_mask[label_img]
        label_img[remove]=0
        labels=np.unique(label_img)
        label_img=np.searchsorted(labels,label_img)
        obj=scipy.ndimage.find_objects(label_img)
        pos=np.zeros((len(obj),2),int)
        for i in range(len(obj)):
            p=np.unravel_index(np.argmax(img_data[obj[i][0],obj[i][1]]),dims=img_data[obj[i][0],obj[i][1]].shape)
            p=p[0]+obj[i][0].start,p[1]+obj[i][1].start
            pos[i,:]=p
        num_peaks=pos.shape[0]
        sig-=1
    if(num_peaks<min_peaks):    
        raise ValueError('Minumum significance level has been reached')  
    try:
        return scipy.stats.theilslopes(pos[:,0],pos[:,1])
    except IndexError:
        line=scipy.stats.theilslopes(pos[:,1],pos[:,0])
        line[0]=np.inf
        return line
    
'''
Reduce a single image. 

Arguments:
    img_data: 2D array containing intensity values
    line: an array-like object containing values at indices:
        0: slope
        1: y-intercept
    intensities will be taken along the line if provided. 

Returns:
    1D array containing reduced 1D intensity profile
'''
def reduce_data(img_data,line=None,percentile=60):
    num_points=img_data.shape[0]
    if line is None:
        line=find_line(img_data,percentile=percentile)
        yint=line[1]
        slope=line[0]
        if np.isinf(slope):
            x0=yint
            x1=yint
            y0=0
            y1=img_data.shape[1]
        elif yint<0:
            y0=0 
            x0=-yint/slope
        elif yint>img_data.shape[1]-1:
            y0=img_data.shape[1]-1
            x0=y0/slope-yint/slope
        else:
            x0=0
            y0=yint
        out_bound=slope*img_data.shape[0]-1+yint
        if not np.isinf(out_bound) and np.abs(slope)>1 and (out_bound<0 or out_bound>img_data.shape[0]-1):
            y1=(img_data.shape[0]-1)*(out_bound>img_data.shape[0]-1)
            x1=y1/slope-yint/slope
        elif not np.isinf(out_bound):
            x1=img_data.shape[1]-1
            y1=slope*img_data.shape[1]-1+yint
        x,y=np.linspace(x0,x1,num_points),np.linspace(y0,y1,num_points)
        return scipy.ndimage.map_coordinates(img_data,np.vstack((y,x)))

    else:
        yint=line[1]
        slope=line[0]
        if np.isinf(slope):
            x0=yint
            x1=yint
            y0=0
            y1=img_data.shape[1]
        elif yint<0:
            y0=0 
            x0=-yint/slope
        elif yint>img_data.shape[1]-1:
            y0=img_data.shape[1]-1
            x0=y0/slope-yint/slope
        else:
            x0=0
            y0=yint
        out_bound=slope*img_data.shape[0]-1+yint
        if not np.isinf(out_bound) and np.abs(slope)>1 and (out_bound<0 or out_bound>img_data.shape[0]-1):
            y1=(img_data.shape[0]-1)*(out_bound>img_data.shape[0]-1)
            x1=y1/slope-yint/slope
        elif not np.isinf(out_bound):
            x1=img_data.shape[1]-1
            y1=slope*img_data.shape[1]-1+yint
        x,y=np.linspace(x0,x1,num_points),np.linspace(y0,y1,num_points)
        return scipy.ndimage.map_coordinates(img_data,np.vstack((y,x)))
        
'''
Reduce a datatset. The 1D profile peaks may be shifted by some amount along a line between each image, so the method attempts to fit the line and shift the intensity profiles such that the resulting array has bragg peak intensities along the same axis.

Arguments:
    prefix: the path to the images, followed by the common prefix (i.e. "C:/Username/Path/To/Files/image_prefix)
    The images are assumed to be .fits files labelled with a hyphen followed by consecutive 5 digit numbers (i.e. -00001, -00002, ...)
    mask_sig: how many median absolute deviations above the median a pixel must be to count as a bragg peak. Used to shift 1D profiles
    
Returns:
    2D array containing 1D intensity profiles for every image in the dataset
'''
def reduce_dataset(prefix,mask_sig=50):
    num_images=0
    img_data=fits.open(prefix+'-00001.fits',ext=2)
    energy=img_data[0].header['Beamline Energy']
    x0=np.arange(img_data[2].data.shape[0])*energy
    while os.path.isfile((prefix+'-%05d.fits'%(num_images+1))):
        num_images+=1
        
    spect=np.zeros((num_images,img_data[2].data.shape[0]))    
    lines=np.zeros((num_images,4))
    
    for i in range(num_images):
        img_data=fits.open(prefix+'-%05d.fits'%(i+1),ext=2)[2].data
        lines[i]=find_line(img_data)
    line=np.median(lines,axis=0)
    
    for i in range(num_images):
        img_data=fits.open(prefix+'-%05d.fits'%(i+1),ext=2)
        y=reduce_data(img_data[2].data,line)
        energy=img_data[0].header['Beamline Energy']
        x=np.arange(np.max(y.shape))*energy
        spect[i,:]=scipy.interpolate.interp1d(x,y,fill_value=0,bounds_error=False)(x0)
    
    
    mask=spect>mask_sig*np.median(spect)
    labels,num_label=scipy.ndimage.measurements.label(mask)
    obj=scipy.ndimage.find_objects(labels)
    pts=np.nonzero(mask[obj[0]])
    slope,yint=scipy.stats.theilslopes(pts[1],pts[0])[:2]
    yint=yint+obj[0][1].start-obj[0][0].start*slope
    
    shift_to=int(round(slope*spect.shape[0]+yint))
    for i in range(spect.shape[0]):
        peak_pos=int(round(i*slope+yint))
        spect[i,:]=np.roll(spect[i,:],shift=shift_to-peak_pos)
    
    return spect
