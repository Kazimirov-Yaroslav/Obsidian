---
title: "SOHO/MDI and SDO/HMI sunspot area measurement and analysis"
source_article: "P. J. Meadows, 2025 - SOHO/MDI and SDO/HMI sunspot area measurement and analysis (RASTI)"
tags:
  - sunspot-area
  - solar-activity
  - soho
  - sdo
  - paper/original
authors: "P. J. Meadows"
year: 2025
journal: "RASTI 4, 1–14"
doi: "10.1093/rasti/rzaf024"
---

# SOHO/MDI and SDO/HMI sunspot area measurement and analysis

**Author:** P. J. Meadows  
**Journal:** RASTI 4, 1–14 (2025)  
**DOI:** [10.1093/rasti/rzaf024](https://doi.org/10.1093/rasti/rzaf024)

---

## Contents

### [[#1. Introduction|1. Introduction]]
### [[#2. Images and Measurement Approach|2. Images and Measurement Approach]]
### [[#3. Calibration|3. Calibration]]
### [[#4. Time Series|4. Time Series]]
### [[#5. Monthly Trends|5. Monthly Trends]]
### [[#6. Comparison with Kislovodsk during Cycle 25|6. Comparison with Kislovodsk during Cycle 25]]
### [[#7. Sunspot Group Areas|7. Sunspot Group Areas]]
### [[#8. Conclusions|8. Conclusions]]
### [[#Appendix A: Projected and Corrected Areas|Appendix A: Projected and Corrected Areas]]
### [[#Appendix B: Disc Edge, Position, and Radius|Appendix B: Disc Edge, Position, and Radius]]
### [[#Appendix C: Detailed Algorithm|Appendix C: Detailed Algorithm]]
### [[#Appendix D: SOHO and SDO Comparison|Appendix D: SOHO and SDO Comparison]]

---

## Abstract

Sunspot area is a key indicator of solar activity, providing an alternative to traditional empirical sunspot number. This paper presents a new catalogue of total sunspot area derived from full-disc white-light images acquired by the SOHO Michelson Doppler Imager (MDI) and the SDO Helioseismic and Magnetic Imager (HMI) from 1996 to the present. The methodology involves a multistep threshold approach to identify sunspots which has been calibrated against the Debrecen Photoheliographic Data (DPD) for consistency with previous sunspot area catalogues. The data set includes daily sunspot area measurements, with projected and corrected values, spanning solar Cycles 23, 24, and 25. Long-term trends in sunspot area are analysed, including hemispheric asymmetries and the relationship with the International Sunspot Number. Significant changes in sunspot area, including extreme daily variations and cycle maxima, are identified. Additionally, the study includes a comparison, during Cycle 25, with area measurements from the Kislovodsk solar station of the Pulkovo observatory. Finally, example high-cadence sunspot group area evolution is shown for four large sunspot groups, as is the different split of sunspots into groups by several observatories.

**Key words:** Data Methods – Sunspot Area – Catalogue.

---

## 1. Introduction

The size of sunspots is a physical measure of solar activity in white-light. Unlike the usual measure of white-light solar activity, the sunspot number or the number of sunspot groups, sunspot area is not dependent on the separation of sunspots into groups by a solar observer. This grouping process can be challenging, especially during periods of moderate or high solar activity.

Both sunspot area and sunspot number measurements can be affected by the quality of drawings or images, particularly due to atmospheric seeing conditions. However, this is not an issue for satellite-based white-light images, such as those acquired by the Solar and Heliospheric Observatory (SOHO) (Scherrer et al. 1995) and the Solar Dynamics Observatory (SDO) (Pesnell, Thompson & Chamberlin 2011; Hoeksema et al. 2018).

The Royal Greenwich Observatory (RGO) sunspot catalogue, which includes sunspot group position and area measurements, began in 1874 May and continued until the end of 1976. These measurements were obtained from photographic plates taken by RGO and its partner observatories in Cape Town (South Africa), Kodaikanal (India), and Mauritius (Hathaway 2015). Both umbral and total sunspot areas (including penumbra and umbra) were measured and corrected for foreshortening. Corrected sunspot areas were recorded in units of millionths of the Sun's visible hemisphere (μHem).

In 1976, the programme of daily solar observations was transferred to the Debrecen Observatory of the Hungarian Academy of Sciences (Baranyi, Győri & Ludmány 2016; Győri, Ludmány & Baranyi 2017). The final version of the Debrecen catalogue covers data up to and including 2015. The only other sunspot catalogue currently providing daily area measurements is that from the Kislovodsk solar station of the Pulkovo Observatory, Russia. Note that although the USAF/NOAA Solar Region Reports (SRS) provide sunspot group areas, these are incorrectly calculated and can be underestimated by up to 50 per cent (Meadows 2020). Mandal et al. (2020) amalgamated the Greenwich, Debrecen, and Kislovodsk catalogues.

The Greenwich, Debrecen, Kislovodsk, and Mandal sunspot catalogues are all available online.

The purpose of this paper is to create a sunspot catalogue of full-disc sunspot areas using continuum images from the SOHO Michelson Doppler Imager (MDI) and the SDO Helioseismic and Magnetic Imager (HMI). This catalogue covers the period from 1996 to the present. The full-disc area is referred to here as the Total Sunspot Area, with two types of area calculated: projected and corrected, as described in Appendix A. The SOHO MDI continuum image is derived from filtergrams near the Ni-I photospheric spectral line at 676.8 nm (Berner & Domingo 1995). The SDO HMI instrument, an enhanced version of SOHO MDI, obtains continuum images near the Fe I 617.3 nm spectral line (Scherrer et al. 2012).

To ensure consistency between this new catalogue and existing ones, two steps have been taken: first, the number of pixels in the SOHO and SDO images has been kept similar to those used previously; secondly, areas measured in SOHO and SDO images have been compared with Debrecen catalogue data from Solar Cycles 23 and 24. Greenwich sunspot areas were measured using a specially constructed large position micrometer. The micrometer had a diaphragm ruled with squares, each one-hundredth of an inch wide, which were used to count the number of squares occupied by sunspots (Maunder 1921; Meadows 2020). Given that the Greenwich photographic plates had a diameter of 10 inches, this corresponds to a digital scanned image with a diameter of approximately 1000 pixels, comparable to the SOHO 1024 and SDO 1k image sizes. Sunspot areas in SOHO and SDO images are measured using a multistep threshold approach to create a sunspot mask. The threshold value has been calibrated against Debrecen catalogue areas to minimize area differences, with additional visual check of the mask against the original SOHO and SDO image.

Several techniques have been developed to distinguish sunspots (umbra and penumbra) from the photosphere in white-light solar images, each using different image processing strategies. Adaptive thresholding, as employed by Hanaoka (2022), adjusts the intensity threshold locally across the image to accommodate varying illumination and contrast, enabling more robust detection of sunspots for non-uniform images. Győri (2015) implemented a contour-based method that utilizes set and graph operations to delineate sunspot boundaries. Cho et al. (2015) combined global thresholding with region-growing techniques, where initial seed regions identified by intensity criteria are expanded based on spatial connectivity and pixel similarity. Sayez et al. (2023) applied segmentation and clustering methods within a deep learning framework, specifically a convolutional neural network (CNN) trained to identify, group, and classify sunspot regions from ground-based white-light images.

Finally, mathematical morphology was employed in the works of Pucha, Hiremath & Gurumath (2016) and Bourgeois et al. (2024). Pucha et al. (2016) used classical morphological operations such as dilation, erosion, and morphological opening to isolate sunspot structures. Bourgeois et al. (2024) extended this approach by refining structuring elements and incorporating multiscale morphological analysis to enhance the separation of sunspot umbra and penumbra, improving detection accuracy even in the presence of limb darkening and instrumental artefacts.

Recent advancements have introduced deep learning techniques for sunspot detection. Mourato, Faria & Ventura (2024) implemented both semantic segmentation and instance segmentation approaches using U-Net and Mask R-CNN architectures, respectively. Chen et al. (2025) proposed a bias-free deep learning approach utilizing a convolutional neural network trained on a diverse data set of solar images.

The multistep threshold approach used here differs from these techniques. It relies on the fact that the flattened images have an almost uniform background and so the approach presented in this paper may not be suitable for all ground-based white-light images.

The structure of this paper is as follows: (i) Section 2 describes how SOHO and SDO images are obtained, how the centre and radius of the solar disc are determined, the normalization process to ensure pixel value consistency over time, and the methodology for creating the sunspot mask. (ii) Section 3 explains the calibration of the multistep threshold approach. (iii) Section 4 presents the SOHO and SDO total sunspot area time series from 1996 onwards. It also identifies dates with the greatest daily area changes and periods of peak sunspot activity. (iv) Section 5 discusses long-term trends in solar activity, typically represented by monthly averages. It includes asymmetries in sunspot distribution between the northern and southern hemispheres, as well as the eastern and western hemispheres, and examines the relationship between total sunspot area and sunspot number. (v) Section 6 compares SDO sunspot areas with those from Kislovodsk, currently the only provider of daily and group sunspot area measurements, during Cycle 25. (vi) Section 7 analyses short-term variations in sunspot area, focusing on the largest sunspot groups from Cycles 23, 24, and 25, taking advantage of the short time interval between SOHO and SDO images. (vii) Section 8 presents the conclusions of this paper.

## 2. Images and Measurement Approach

The SOHO MDI 1024 × 1024 flattened continuum images were downloaded from the ESA SOHO website. These have filenames of the form yyyymmdd hhmm mdiigr 1024.jpg. The advantage of using flattened images is that the variation in intensity between the centre of the disc and the limb has already been removed. There are typically four SOHO continuum images per day.

SDO HMI images were obtained from the Joint Science Operations Center website by selecting the flattened 1024x1024 continuum images with filenames of the form yyyymmdd hhmmss Ic flat 1k.jpg. These images are typically available every 15 min.

Both the SOHO and SDO images are JPEG versions of the original FITS files and have been used for convenience in terms of their reduced size and ease of access.

Two SOHO and SDO data sets have been downloaded. The first corresponds to the date and time of daily sunspot areas from the Debrecen catalogue for use in deriving sunspot mask threshold values and are referred to as the calibration data set. The second is for daily sunspot area measurements for which images closest to 0 h and 12 h UT are used: the time-series data set.

For SOHO there are 1096 calibration data set images for the three years from 2000 to 2002 (corresponding the high solar activity during Cycle 23) and 8834 time-series images from 1996 May to 2009 December. For SDO there are 1095 calibration data set images for the three years from 2013 to 2015 (high Cycle 24 activity) and 10 867 time-series images from 2010 May up to the end of 2025 March.

For both SOHO and SDO images, a greyscale version is used for deriving the measurement of sunspot and faculae area. All subsequent reference to pixel values in this paper refers to the greyscale values.

The procedure to determine the position and size of the solar disc within the image is described in Appendix B. It is crucial to have an accurate fit to the solar disc, especially for the corrected sunspot area near the solar limb.

Since the method for identifying sunspots relies on a threshold approach, any variation in the mean brightness of SOHO and SDO images must be accounted for. This is achieved by calculating the mean pixel intensity in two image regions that are free of sunspots and faculae. These regions are: (i) between latitudes of 45° & 60°N and 30°E & 30°W of the central meridian and (ii) the corresponding region in the southern hemisphere.

The pixel coordinates of these regions will vary throughout the year through the change in the heliographic latitude of the centre of the disc (B₀).

![[1.jpg|Figure 1]]
**Figure 1:** Background region mean pixel value before and after normalization for SOHO (top) and SDO (bottom).

Fig. 1 shows the mean pixel value for the time-series data set based on the average of the two regions described above. The blue points are the original mean pixel values while the superimposed red points are from the normalized images.

The normalization factor used to create a new image, with integer pixel values, has been derived from the mean pixel value from the time-series data sets for the years 2000 to 2002 inclusive for SOHO and 2013 to 2015 inclusive for SDO. The multiplicative normalization factors are:
- (i) 1/185.74 for SOHO
- (ii) 1/201.81 for SDO

The mean pixel values of most images remain unchanged after applying the normalization factor. However, 3.7 per cent of SOHO time-series images and 14.8 per cent of SDO images (up to the end of 2025 March) are affected by the normalization. Most of the affected SDO images were acquired before the end of February 2013. The same normalization is applied to all data sets and has been used for all subsequent measurements and analyses.

The two data sets were checked for image quality issues, which were removed and replaced where possible. Identified issues include:
- (i) blank discs
- (ii) missing segments
- (iii) significant reductions in overall image intensity
- (iv) defocused images
- (v) intensity gradients across the disc.

These problems were more common in SOHO images. The standard deviation of pixel values in the background regions described earlier was effective in identifying many of the problem images.

All images were visually inspected, and replacements found where possible.

The overall approach to determining whether a solar disc pixel corresponds to a sunspot (penumbra or umbra) or faculae involves applying a pixel value threshold to create a mask. These masks are then used to calculate the areas of sunspots (penumbra and umbra), umbrae, or faculae. As described in Appendix C, several steps are required to correctly identify sunspot pixels and exclude similarly valued pixels elsewhere in the photosphere.

Although SOHO images are available in 2010 and 2011, which includes a overlap with SDO from 2010 May, all the 2010 and 2011 SOHO images are of poorer quality as explained in Appendix D. Consequently, they have not been included in the time-series data set. However, they have used to assess the impact of the different resolutions of the SOHO/MDI and SHO/HMI instruments as shown in Appendix D.

## 3. Calibration

A key requirement for any long-duration time series is consistency with previous time-series data. For the time period covered by SOHO and SDO images, the reference sunspot area data base is the Debrecen Photoheliographic Data (DPD). Therefore, the threshold values for distinguishing sunspots (penumbra and umbra) from the photosphere, and for the full solar disc, has been calculated by a comparison with DPD sunspot projected and corrected area data available daily for the whole disc. In addition, a visual check was performed between the sunspot mask regions and the original SOHO and SDO images.

DPD data from the three years 2000 to 2002, which includes periods of high activity in Cycle 23, were compared with sunspot areas derived from SOHO images, while data from the years 2013 to 2015, a period of high activity in Cycle 24, were compared with sunspot areas derived from SDO images.

The SOHO and SDO images closest in date and time to the DPD total sunspot area data were selected to minimize the impact of sunspot area variations throughout each day. The mean temporal difference between SOHO images and DPD data was −3.2 ± 130.7 min (with the largest difference being −12.5 h due to data unavailability), while for SDO images, the temporal difference was −0.4 ± 12.6 min (with the largest difference being 5.5 h).

Fig. 2 shows the comparison between SOHO and DPD daily projected and corrected areas for a sunspot pixel threshold value of 165. The red line is the 1:1 relationship. A similar plot for SDO images is shown in Fig. 3 for a threshold of 184. The projected area standard deviation for SOHO, Fig. 1 top, is 194 μDisc while for SDO it is 116 μDisc. The corresponding values for corrected area are 196 μHem and 100 μHem, respectively. The reduced time interval between DPD and SDO images, compared to SOHO images, likely contributes to the smaller standard deviations for SDO.

![[2.jpg|Figure 2]]
**Figure 2:** Comparison of SOHO and DPD total projected (top) and corrected areas (bottom) from 2000 to 2002.

![[3.jpg|Figure 3]]
**Figure 3:** Comparison of SDO and DPD total projected (top) and corrected areas (bottom) from 2013 to 2015.

Tables 1 and 2 present the relationship between SOHO and SDO with DPD total areas in terms of slope (Figs 2 and 3), absolute difference, and ratio for both projected and corrected total areas. The bold values are the closest to either unity or zero. The slope values were derived from the average of the slope for SOHO or SDO versus DPD and the reciprocal of DPD versus SOHO or SDO.

**Table 1:** SOHO/MDI and DPD area comparison.

| Threshold | Proj. Slope | Corr. Slope | Proj. Diff. | Corr. Diff. | Proj. Ratio | Corr. Ratio |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| 164 | 0.979 | 0.979 | 94.8 | 69.5 | 1.073 | 1.080 |
| 165 | 0.995 | 0.995 | 47.3 | 34.9 | 1.053 | 1.060 |
| 166 | 1.012 | 1.011 | −1.8 | −0.7 | 1.034 | 1.041 |

**Table 2:** SDO/MDI and DPD area comparison.

| Threshold | Proj. Slope | Corr. Slope | Proj. Diff. | Corr. Diff. | Proj. Ratio | Corr. Ratio |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| 183 | 0.988 | 0.999 | 31.9 | 5.8 | 0.967 | 0.989 |
| 184 | 1.004 | 1.015 | 5.4 | −14.0 | 0.987 | 1.010 |
| 185 | 1.022 | 1.033 | −24.5 | −35.5 | 1.011 | 1.034 |

Finally, a visual comparison was performed using a large sunspot group from Cycles 23 and 24, applying three different threshold values.

Figs 4 and 5 show the corresponding masks for these three threshold values for the largest sunspot groups from Cycle 23 (SOHO) and Cycle 24 (SDO), with the corrected area for each mask also indicated. Note that the middle mask threshold was selected based on the earlier comparison with DPD data, and the visual comparison confirms the appropriateness of this choice.

![[4.jpg|Figure 4]]
**Figure 4:** AR9393 from 2001 March 29 at 00:00 UT showing original SOHO image and three masks with different thresholds (corrected areas in brackets).

![[5.jpg|Figure 5]]
**Figure 5:** AR12192 from 2014 October 24 at 13:30 UT showing original SDO image and three masks with different thresholds (corrected areas in brackets).

## 4. Time Series

Figs 6 and 7 show the corrected total area derived from SOHO and SDO imagery, respectively.

![[6.jpg|Figure 6]]
**Figure 6:** SOHO total corrected sunspot area.

![[7.jpg|Figure 7]]
**Figure 7:** SDO total corrected sunspot area.

Table 3 presents the change in total corrected area greater than 1750 μHem, based on both the SOHO and SDO time-series data set, along with an explanation for the change – namely, the appearance or disappearance of a sunspot group onto or off the disc. These changes are determined from the difference in total disc area recorded in the first image of each day, adjusted for a 1-d interval.

**Table 3:** Greatest change in SOHO and SDO total corrected areas.

| Date | Daily change in Corrected Area (μHem) | Comment |
|------|--------------------------------------|---------|
| 2003 Nov 04 | −3900 | Disappearance of AR 10486 & 10488 |
| 2002 Aug 11 | 2300 | Appearance of AR 10069 |
| 2014 Oct 30 | −2110 | Disappearance of AR 12192 |
| 2001 Oct 24 | 1990 | Appearance of AR 9682 |
| 2003 Oct 23 | 1990 | Appearance of AR 10486 |
| 2014 Oct 17 | 1980 | Appearance of AR 12192 |
| 2024 Aug 04 | 1970 | Appearance of AR 13780 |
| 2002 Aug 23 | −1960 | Disappearance of AR 10069 |
| 2000 Sep 17 | 1940 | Appearance of AR 9167 |
| 2024 Oct 01 | 1910 | Appearance of AR 13848 |
| 2000 Apr 20 | 1850 | Appearance of AR 8970 |
| 2024 Mar 02 | −1820 | Disappearance of AR 13590 |

The dates for which the daily corrected area exceeded 5000 μHem are given in Table 4 based on the entire time-series data set. The date of the largest total disc area is given in brackets. Additionally, this table includes the main groups along with their USAF/NOAA AR number, McIntosh group classification (McIntosh 1990), and corrected area, as measured from SOHO or SDO images.

**Table 4:** SOHO and SDO total corrected areas greater than 5000 μHem.

| Date(s) | Max. Daily Area (μHem) | Largest Group (McIntosh type, Area (μHem)) |
|---------|----------------------|-------------------------------------------|
| 2024 Aug 04–Aug 09 (Aug 05) | 6050 | AR 13780 (Fkc, 2790) |
| 2003 Oct 23–Nov 03 (Oct 29) | 8340 | AR 10486 (Fkc, 3510) |
| 2001 Mar 27–Apr 03 (Mar 29) | 6590 | AR 9393 (Fki, 3470) |
| 2024 Oct 04–Oct 07 (Oct 04) | 5980 | AR 13842 (Ekc, 1460) |
| 2000 May 15–May 16 (May 15) | 5590 | AR 8996 (Ekc, 1910) |
| 2024 Jul 31–Aug 01 (Jul 31) | 5580 | AR 13765/66/67 (Fkc, 1670) |
| 2014 Oct 22–Oct 24 (Oct 24) | 5480 | AR 12192 (Fkc, 4600) |
| 2024 Jun 21–Jun 22 (Jun 21) | 5330 | AR 13712 (Ekc, 1750) |
| 2000 Jul 11 | 5240 | AR 9077 (Fki, 1690) |
| 2001 Sep 24–Sep 26 (Sep 25) | 5230 | AR 9628 (Fkc, 1610) |
| 2002 Aug 19 | 5100 | AR 10069 (Ekc, 2820) |

The highest daily corrected area of 8340 μHem on 2003 Oct 29, due to the presence of AR 10484 near the western limb and AR 10486 together with AR 10488 near the central meridian.

## 5. Monthly Trends

As with the International Sunspot Number (Ri), long-term area trends can be assessed by taking monthly averages (Mean Daily Frequency, MDF), in particular smoothed MDFs, which are calculated using a 13-month average as given by:

$$MDF_S = \frac{1}{5} \sum_{i=-5}^{5} MDF_i + \frac{1}{2}(MDF_{-6} + MDF_6) \quad (1)$$

Fig. 8 shows the combined SOHO and SDO smoothed corrected area MDF for the total disc, the northern hemisphere, and the southern hemisphere, with the unsmoothed MDF shown in the background.

![[8.jpg|Figure 8]]
**Figure 8:** Total and north and south corrected area mean daily frequency.

As with Ri, the double-peaked structure of Cycles 23 and 24 is also reflected in total disc sunspot areas, caused by asymmetry in activity between the northern and southern hemispheres during these cycles.

In Cycle 25, the later months of 2024 also exhibited asymmetric activity between the northern and southern hemispheres.

The dates of maxima and minima of the smoothed area MDF of the last three cycles are given in Table 5 along with the smoothed total corrected area. Note that for Cycle 23 minima, the availability of imagery means the earliest SOHO MDF data is from 1996 June, so the actual minimum may have occurred earlier. For Cycle 25, the latest smoothed area data is from 2024 September, meaning the Cycle 25 maximum may be higher than indicated in this table. For comparison, Table 6 provides the corresponding Ri minima and maxima. With the exception of the Cycle 23 minimum, the differences between the area and sunspot number cycle minima and maxima are within two months of each other.

**Table 5:** Smoothed total corrected areas minima and maxima.

| Cycle | Minima Date | Minima Corr. Area (μHem) | Maxima Date | Maxima Corr. Area (μHem) |
|-------|-------------|-------------------------|-------------|-------------------------|
| 23 | 1997 Feb | 98 | 2002 Feb | 2424 |
| 24 | 2008 Nov | 3 | 2014 Jun | 1512 |
| 25 | 2019 Dec | 6 | 2024 Aug | 2324 |

**Table 6:** Smoothed international sunspot number minima and maxima.

| Cycle | Minima Date | Minima Ri | Maxima Date | Maxima Ri |
|-------|-------------|-----------|-------------|-----------|
| 23 | 1996 Aug | 11.2 | 2001 Nov | 180.3 |
| 24 | 2008 Dec | 2.2 | 2014 Apr | 116.4 |
| 25 | 2019 Dec | 1.8 | 2024 Sep | 159.3 |

Fig. 9 shows the corrected area MDF between the eastern and western hemispheres split between the northern and southern hemispheres (the split is necessary due to the asymmetry between these hemispheres during a solar cycle). The solid curves show the smoothed MDFs while the unsmoothed MDF is shown the background. Based on the smoothed MDF, there is very little difference, as expected, between the eastern and western hemispheres.

![[9.jpg|Figure 9]]
**Figure 9:** North east and north west (top) and south east and south west (bottom) corrected area MDF.

Numerically, the area asymmetry between the hemispheres can be quantified by the following expression, for the northern (MDF_North) and southern (MDF_South) hemispheres by:

$$Asymmetry = \frac{(MDF_{North} - MDF_{South})}{(MDF_{North} + MDF_{South})} \quad (2)$$

where an asymmetry value of 0 or close to 0 indicate no or very little asymmetry.

Table 7 presents the asymmetry values for Cycles 23, 24, and 25 (up to the end of 2025 March) between the northern and southern hemispheres and between the eastern and western hemispheres in the northern and southern hemispheres. This analysis shows that, when considering an entire cycle, there is very little overall asymmetry, although north-south asymmetry does occur during individual cycles.

The asymmetry between the eastern and western hemispheres within both the northern and southern hemispheres is relatively small, even during solar cycles.

**Table 7:** Cycle smoothed area asymmetry.

| Cycle | North and south | North-east and north-west | South-east and south-west |
|-------|-----------------|--------------------------|--------------------------|
| 23 | −0.068 | 0.022 | −0.018 |
| 24 | −0.025 | 0.046 | −0.002 |
| 25 | −0.148 | −0.008 | −0.025 |

Finally, Fig. 10 shows the relationship between the smoothed Ri and the smoothed corrected area, based on all SOHO and SDO smoothed MDF values (up to the end of 2025 March). The plot is also divided into the ascending and descending phases of each cycle.

A strong correlation can be observed between the two quantities: the red curve represents a quadratic fit based on all data (R² = 0.983), given by the following equation:

$$Ri_S = -9.267 \times 10^{-6} MDF_S^2 + 0.09738 MDF_S \quad (3)$$

![[10.jpg|Figure 10]]
**Figure 10:** Relationship between smoothed total area MDF and smoothed international sunspot number Ri.

A similar plot to Fig. 10 appears in Hathaway (2015), using RGO Sunspot Areas. However, that analysis was based on Ri version 1, which is on average 1.62 ± 0.14 times smaller than the latest Ri version 2 (Clette et al. (2014)) over the period 1749–2014 (when version 1 was replaced). For corrected areas above approximately 500 μHem in Fig. 10, a separation is observed between the ascending and descending phases of the cycles. This suggests that for a given sunspot number, the area of most sunspot groups after solar maximum is larger than before.

## 6. Comparison with Kislovodsk during Cycle 25

Since the only sunspot catalogue that currently contains sunspot area measurements is from Kislovodsk, a comparison has been conducted between these measurements and the areas derived here from SDO imagery.

This comparison covers Cycle 25 from 2022 up to the end of 2025 March. Fig. 11 presents a comparison between SDO and Kislovodsk total projected and corrected areas during periods of moderate and high activity. Along with Table 8 (first row), it can be seen that Kislovodsk area measurements tend to underestimate compared to SDO areas.

![[11.jpg|Figure 11]]
**Figure 11:** Comparison of SDO and Kislovodsk total projected (top) and corrected areas (bottom) from 2022 to up to the end of 2025 March.

**Table 8:** SDO and Kislovodsk area comparison during cycle 25.

| Year(s) | Proj. Slope | Corr. Slope | Proj. Diff. | Corr. Diff. | Proj. Ratio | Corr. Ratio |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|
| 2022–25 | 1.044 | 1.060 | −78.9 | −0.85 | 0.973 | 0.952 |
| SDO | 1.067 | 1.053 | −130.9 | −87.1 | 0.951 | 0.953 |

Many of the Kislovodsk measurements are made using SDO images, presumably when cloudy conditions prevent their own observations from being made. Fig. 12 and Table 8 (second row) are just for Kislovodsk SDO (KSDO) measurements. Again, there is an underestimation of the KSDO area measurements compared to the SDO areas but there are many instances where there is good agreement.

![[12.jpg|Figure 12]]
**Figure 12:** Comparison of SDO and Kislovodsk SDO total projected (top) and corrected areas (bottom) from 2022 up to the end of 2025 March.

An example of the difference between Kislovodsk and SDO is illustrated in Fig. 13 for the largest sunspot group of Cycle 25 (up to the end of 2025 March) – AR 13364, which crossed the central meridian on 2024 May 7. Also shown are instances where Kislovodsk used SDO images. During the initial rise in area, there is a clear underestimation of corrected KSDO area measurements by several hundred μHem, even when SDO images were used.

The largest differences between projected and corrected areas between KSDO and SDO measurements occur when there are complex sunspots groups on the disc which contain many sunspots. However, measurement of individual sunspots from SDO and data files associated with each KSDO image show good agreement for single, isolated sunspots (differences of < 10 μHem) but not for larger more complex sunspots groups.

Note that in the latter part of AR 13364's passage across the disc in Fig. 13, the number of sunspots within the group decreased. By the time it neared the western limb, only a few individual sunspots remained, leading to better agreement between KSDO and SDO, particularly on 2024 May 10 and 12.

![[13.jpg|Figure 13]]
**Figure 13:** Projected (top) and Corrected area (bottom) comparison between SDO and Kislovodsk for AR 13664 during 2024 May.

## 7. Sunspot Group Areas

In addition to calculating the total area of all sunspots on the disc, the masks can be used to determine the area of individual sunspot groups. The approach for grouping sunspots involves calculating the difference in longitude and latitude between each pair of sunspots. If the separation falls within an ellipse measuring 10° in longitude and 3° in latitude, the sunspots are assigned to the same group. This procedure is repeated for all sunspot pairs until every sunspot is allocated to a group.

Before applying this grouping process, sunspots are first assigned to either the Northern or Southern hemisphere, ensuring that a group cannot contain sunspots from both hemispheres. Finally, only groups containing more than one pixel and with a corrected area greater than 5 μHem are included in the automatic group identification process.

It is unlikely that such small groups would be observed by ground-based observatories contributing to sunspot area catalogues or the International Sunspot Number Ri.

Occasionally, the automatic procedure incorrectly assigns sunspots to a single group when they would normally be split into two nearby groups. In such cases, manual intervention is required by placing a latitude/longitude boundary between the two sets of sunspots and re-running the group identification software. It should be noted that the identification of sunspot groups here is based purely on visual appearance and the author's experience. As a result, the grouping may differ from the active regions (ARs) designated in the USAF/NOAA solar region reports.

An example of sunspot group identification output is shown in Fig. 14 and Table 9 for 2014 October 24, which includes the large sunspot group AR 12192 from Cycle 24. The group identification in Table 9 follows an ordering system based on whether the group is in the Northern or Southern hemisphere and its position from west to east within that hemisphere. The heliographic latitude and longitude are determined from the centroid position of all sunspots within the group. This table also gives the projected and corrected area as well as the number of sunspots and the AR number for each group.

![[14.jpg|Figure 14]]
**Figure 14:** Sunspot separation into groups for SDO image from 2014 October 24 at 13:30 UT. The annotation below each group gives the group identification, McIntosh group classification, corrected area in μHem, and the USAF/NOAA AR number.

**Table 9:** Sunspot groups for SDO on 2014 October 24 at 13:30 UT.

| Group | Lat. (°) | Long. (°) | Proj. Area (μDisc) | Corr. Area (μHem) | Num. Spots | AR ID |
|-------|----------|-----------|-------------------|-------------------|------------|-------|
| N1 | 5.3 | 287.7 | 187 | 159 | 3 | 12193 |
| S1 | −13.6 | 247.2 | 8371 | 4525 | 107 | 12192 |
| S2 | −11.7 | 208.1 | 224 | 129 | 4 | 12194 |
| N2 | 8.0 | 180.0 | 201 | 165 | 11 | 12195 |
| S3 | −12.3 | 164.7 | 10 | 15 | 1 | 12197 |
| S4 | −2.9 | 160.1 | 29 | 53 | 1 | 12196 |

By identifying individual sunspot groups, it is possible to track the evolution of their area across the solar disc with high cadence – up to four images per day for SOHO and every 15 min for SDO. Fig. 15 show the corrected area of the largest sunspot groups from Cycles 23 (AR 9393 and AR 10486), Cycle 24 (AR 12192), and Cycle 25 (up to the end of 2025 March: AR 13664) during their passage across the solar disc. Such evolution plots could be used to investigate short-term variations in sunspot area and their relationship with other phenomena, such as solar flares. Note that AR 12192 was the largest sunspot group since the five large groups observed between 1946 and 1951 during Cycle 19 (Meadows 2021).

![[15.jpg|Figure 15]]
**Figure 15:** Corrected sunspot area evolution for largest sunspots groups during Cycles 23 (AR 9393 and AR 10486), 24 (AR 12192) and 25 (AR 13664).

The separation of sunspots into groups can depend on the observatory and the observer. Fig. 16 shows an example for AR 13364 from 2014 May 9. Drawings are shown from the Uccle Solar Equatorial Table (USET), located at the Royal Observatory of Belgium (ROB), Specola Solare Ticinese in Locarno, Switzerland, and Mt Wilson, USA are shown alongside the corresponding SDO image for comparison. The USET drawing indicates two groups: the leading (westernmost) sunspot being of type C, while the remaining following sunspot as a separate F type group. The Locarno drawing also indicates two groups but separated in the middle; whereas for the Mt Wilson drawing shows a single group (R and V indicate positive and the negative polarities; Pevtsov et al. 2019). The USAF/NOAA Solar Region Report similarly classifies the sunspots as a single group, whereas the Kislovodsk sunspot catalogue identifies two groups, but with the following (easternmost) sunspots forming a separate group.

![[16.jpg|Figure 16]]
**Figure 16:** AR 13364 from 2024 May 9 from Royal Observatory of Belgium, Specola Solare Ticinese in Locarno, SDO and Mt Wilson. For the drawing annotation to be legible, the orientation is mirrored compared to all the other SOHO and SDO images in this paper.

## 8. Conclusions

This paper presents a catalogue of full-disc sunspot area measurements derived from SOHO Michelson Doppler Imager (MDI) and SDO Helioseismic and Magnetic Imager (HMI) continuum images, spanning the period from 1996 to the present. The total sunspot area, calculated using a multistep threshold method, has been calibrated against the Debrecen Photoheliographic Data (DPD) to ensure consistency with previous sunspot area catalogues. This has been supported by visual confirmation of the successfully derived sunspot masks.

The generation of the area catalogue has been derived by using the multistep threshold method presented here. This method is straightforward to implement, requires no prior knowledge of sunspot locations, performs a normalization to correct for any image to image intensity variations, and removes spurious small sunspots caused by fluctuations in the intensity of the photosphere.

A detailed analysis of sunspot area trends reveals significant asymmetries in solar activity between hemispheres during cycles and highlights major sunspot groups responsible for the largest variations in total sunspot area. The smoothed monthly mean sunspot area follows the expected solar cycle variations and aligns closely with the International Sunspot Number (Ri), reinforcing the validity of the derived area data set. The study also identifies the relationship between sunspot number and sunspot area, indicating that sunspot groups tend to be larger in the declining phase of the solar cycle compared to the ascending phase.

A comparison with the Kislovodsk sunspot catalogue during Cycle 25 reveals a systematic underestimation of sunspot areas in the Kislovodsk data set. This discrepancy is most pronounced for complex sunspot groups, suggesting the need for further refinement in sunspot detection techniques used in the Kislovodsk data set.

Finally, high-cadence area evolution has been presented for a selection of large sunspot groups. It has been shown that for the largest sunspot group of Cycle 25 (up to end 2025 March), AR 13364, its sunspots have been assigned to one or two groups by different observatories. Where two groups have been assigned, the separation of sunspots is different in each of the three cases examined. Ideally, there should be a consensus amongst solar observatories and their solar observers as to which sunspots should be split into groups.

This issue is a particular problem for the calculation of the sunspot number, especially during times of moderate to high solar activity.

## Acknowledgements

The referees are thanked for their comments. This paper has used several sources of publicly available solar imagery and data. The SOHO/MDI images are courtesy of NASA & ESA. The SDO/HMI continuum images are courtesy of NASA/SDO and the HMI science team. The Mt. Wilson drawing is from the the 150-Foot Solar Tower of the Mt. Wilson Observatory. This is operated by UCLA, with funding from NASA, ONR, and NSF, under agreement with the Mt. Wilson Institute. The Locarno sunspot drawing is from the Specola Solare Ticinese, Locarno, Switzerland. The International Sunspot Number data is courtesy of Sunspot Index and Long-term Solar Observations (SILSO), Royal Observatory of Belgium, Brussels.

The creation of the sunspot mask imagery and the calculation of sunspot area were carried out using a Raspberry Pi 5.

## Conflict of Interest

The author declares no conflict of interest.

## Data Availability

The SOHO and SDO total sunspot area catalogue is available on the author's website. The data and mask images underlying this paper will also be shared on a reasonable request to the author.

## References

*References would be listed here*

---

## Appendix A: Projected and Corrected Areas

Two types of sunspot area are calculated: projected and corrected.

The projected area assumes the Sun is a disc of area πR² with the area of all sunspots on the whole disc or of a sunspot group expressed in millionths of the Sun's disc (μDisc). Thus, for a solar image, the projected area, A_P, is given by:

$$A_P = \sum_{i=1}^{n} \frac{10^6}{\pi R^2} = n \frac{10^6}{\pi R^2} = n \cdot A_S \quad (A1)$$

where n is the number of pixels in a sunspot, R is the radius of the disc, and A_S is the projected area of an individual pixel. For SOHO 1024 images, A_S has a value in the range of 1.300 and 1.398 during the year with a mean values of 1.347 while for SDO 1k images, A_S has a value in the range of 1.366 and 1.461 during the year with a mean values of 1.414.

The corrected area has two components: (i) the surface area of a hemisphere of the Sun and (ii) a correction for foreshortening away from the centre of the solar image.

The corrected area, A_C, is expressed in millionths of the Sun's visible hemisphere (μHem) and is given by:

$$A_C = n \frac{10^6}{2\pi R^2} \sum_{i=1}^{n} \frac{1}{\cos(\rho_i)} = n \frac{A_S}{2} \sum_{i=1}^{n} \frac{1}{\cos(\rho_i)} \quad (A2)$$

where _i is the angle on the surface of the Sun from the centre of the image to sunspot pixel i and is given by:

$$\rho = \sin^{-1}\left(\frac{\rho_1}{S}\right) - \rho_1 \quad (A3)$$

where ρ_1 is the angle from the centre of the disc to the sunspot as viewed from Earth and S is the apparent angular radius of the Sun (in degrees) and is given by S(r/R), where r is the distance of the sunspot from the centre of the image in the same units as R, the image disc radius – see Smart (1971) and Duffett-Smith (1988).

Should the heliographic latitude (B) and longitude (L) be known for each sunspot pixel, then alternatively, ρ can be calculated thus:

$$\cos(\rho) = \sin(B_0)\sin(B) + \cos(B_0)\cos(B)\cos(L - L_0) \quad (A4)$$

where B_0 is and L_0 are the heliographic latitude and longitude of the centre of the disc for the date and time of the image (Smart 1971).

The description of the Greenwich corrected area measurement implies, but is not explicitly stated, that a single value of ρ is used per group (presumably in the middle of the group). As indicated above ρ is calculated for each sunspot group pixel and then the correction factor 1/cos(ρ) is summed for all the pixels in a group. For groups consisting of a single symmetrical sunspot, the difference between the two approaches will be small. However, for more complex groups where the leader (westernmost) sunspot is the largest in the group then the Greenwich method would lead to an overestimate of the group's area when it is in the eastern hemisphere and to a underestimate when on the western hemisphere.

## Appendix B: Disc Edge, Position, and Radius

For any sunspot area measurement to be made, the centre of the solar disc and its radius must be accurately determined. This is achieved by: (i) identifying the edge of the disc by using a series of radials extending from the centre of the image to the edge and noting the pixel coordinate with a value closest to half the average pixel value from the central 10 per cent of the image and (ii) using these edge coordinate to perform a least-square fit to a circle.

Before applying these steps, any annotations on the image, such as the date and time in the lower left-hand corner, are removed by setting their pixel values to match the image background.

Figs B1 (top) and B2 (top) show the calculated X and Y coordinates for the centre of the disc. As expected, they are close to the middle coordinates of the image, although some sub-pixel variations are present.

For SDO images, there is a clear improvement in both the X and Y coordinates after 2018 May.

Figs B1 (bottom) and B2 (bottom) display the variation in the solar disc radius, revealing a distinct yearly pattern that reflects the annual change in the satellite–Sun distance. SOHO is located in a halo orbit around the first Lagrangian point (L1), approximately 1.5 million km from Earth towards the Sun. A consequence of the halo orbit is that its distance to the Sun varied by ~0.7 million km (thus leading to the change in the minimum and maximum disc radius from year to year). Also for SOHO, there are a few periods of data unavailability that lead to breaks in the radius plot. SDO is located in a geosynchronous orbit around Earth.

![[17.jpg|Figure B1]]
**Figure B1:** SOHO disc centre position and radius.

![[18.jpg|Figure B2]]
**Figure B2:** SDO disc centre position and radius.

## Appendix C: Detailed Algorithm

For the generation of sunspot, umbrae, and faculae mask images, the SOHO MDI 1024 and SDO HMI flat 1k images are converted to greyscale using the expression:

$$greyscale = nint(0.299r + 0.587g + 0.114b) \quad (C1)$$

where nint is the nearest integer and r, g and b are the red, green and blue pixel values of the original SOHO and SDO images.

### C1 Sunspot (penumbra and umbra)

The following steps are required to create a sunspot mask:

**Step 1 – Apply a threshold.** Search disc pixels and set pixel RGB values to (255, 0, 0), i.e. red, if the pixel value is below the maskThreshold parameter value given in Table C1. Figs C1 and C2 show an example of an original SDO HMI image and the corresponding sunspot mask image after this step – note the numerous spurious isolated mask pixels.

![[19.jpg|Figure C1]]
**Figure C1:** Original image for the region around AR 13780 on 2024 August 7 at 12:00UT.

![[20.jpg|Figure C2]]
**Figure C2:** Step 1 mask for the region around AR 13780 on 2024 August 7.

**Step 2 – Identify and remove isolated mask pixels.** Search for mask pixels within a box of size 3 by 3 pixels – if found check if the surrounding pixels within a 5 by 5 pixel box are also mask pixels. If not, and if there are less than four mask pixels in the 3 by 3 pixel box, set all pixels in the 3 by 3 box to RGB values to (0, 0, 0), i.e. white as shown in Fig. C3. However, also note for the next step if any of the converted pixels within the 3 by 3 box that are less than the extraMaskThreshold parameter value (Table C1) as these could be small sunspots. This step identifies the spurious isolated mask pixels and possible sunspot pixels.

![[21.jpg|Figure C3]]
**Figure C3:** Step 2 mask for the region around AR 13780 on 2024 August 7.

**Step 3 – Remove identified spurious pixels and check for small sunspots.** Here the white pixels are converted into their original greyscale vales. Also perform a pixel check on all identified pixels than could be small sunspots from the previous step, i.e. if these pixels have values less than the maskThreshold parameter value, set them as mask pixels. Note the single background pixels within the mask sunspots in Fig. C4.

![[22.jpg|Figure C4]]
**Figure C4:** Step 3 mask for the region around AR 13780 on 2024 August 7.

**Step 4 – Remove single non-mask pixels within sunspots.** In a similar fashion to removing isolated pixels in step 2, isolated pixels within sunspots are converted to mask pixels, as shown in Fig. C5. This is achieved by searching for non-mask pixels in the middle of a 3 by 3 box where the outer pixels are all mask pixels. The final masked is used for calculating sunspot areas (penumbra and umbra).

![[23.jpg|Figure C5]]
**Figure C5:** Step 4 mask for the region around AR 13780 on 2024 August 7.

### C2 Sunspot umbrae

The creation of the umbra mask is a single threshold step:

**Step 1 – Search within 1 pixel of disc edge for pixel values less than the umbra threshold parameter value, maskUThreshold (Table C1), and assigned to the umbra mask, for example, as shown in Fig. C6.**

![[24.jpg|Figure C6]]
**Figure C6:** Umbra mask for the region around AR 13780 on 2024 August 7.

### C3 Faculae

The following steps are required to create a faculae mask:

**Step 1 – Apply gamma correction.** Enhance the contrast of faculae relative to the photosphere using a gamma correction factor of 0.2.

**Step 2 – Apply a threshold.** Search within 1 pixel of disc edge and assign a faculae mask pixel if the pixel value is greater than maskFThreshold.

**Step 3 – Search for isolated single mask pixels.** Search for mask pixels within a box of size 3 by 3 pixels – if the middle pixel is a mask pixel and all the surrounding pixel are non-mask pixels, convert the middle pixel back to the gamma corrected greyscale value.

**Step 4 – Apply the faculae mask to the original non-gamma-corrected image.**

These processing steps are applied to both SOHO and SDO images using the appropriate thresholds given in Table C1.

**Table C1:** Threshold parameter values.

| Threshold | SOHO MDI | SDO HMI |
|-----------|----------|---------|
| maskThreshold | 165 | 184 |
| extraMaskThreshold | 154 | 170 |
| maskUThreshold | 76 | 110 |
| maskFThreshold | 77 | 111 |

### C4 Composite mask

Once the sunspot, umbrae, and faculae masks have been generated, they can be combined to create a multicoloured composite mask (showing penumbrae, umbrae and faculae). An example SOHO and SDO composite mask is showed in Figs C7 to C9 (Fig. C7 gives to original SOHO image for AR 10486).

This approach produces highly accurate masks for sunspots, umbrae, and faculae, which can be used to derive both projected and corrected areas for the entire solar disc or individual sunspot groups.

All masks are saved in the lossless PNG format. The JPEG format is not used, as it does not preserve the red mask colour (255, 0, 0), which is essential for sunspot pixel identification in software such as the author’s HELIOVIEWER.

![[25.jpg|Figure C7]]
**Figure C7:** Original SOHO image for the region around AR 10486 on 2003 November 1 at 00:00 UT.

![[26.jpg|Figure C8]]
**Figure C8:** Composite SOHO image for the region around AR 10486 on 2003 November 1.

![[27.jpg|Figure C9]]
**Figure C9:** Composite SDO mask image for the region around AR 13780 on 2024 August 7.

## Appendix D: SOHO and SDO Comparison

Both SOHO and SDO images are available during an overlapping period from May 2010 to April 2011, enabling a direct comparison of sunspot areas observed by the two satellites. Unfortunately, SOHO images from the beginning of 2010 exhibit reduced image intensity across the entire solar disc, with further dimming around the limb. The photosphere is also far less uniform. Even after applying the normalization procedure described in Section 2, the multistep threshold approach introduces false mask pixels near the limb, leading to inaccurate area measurements for the full disc. Moreover, a different maskThreshold value is required even for sunspots located away from the limb.

The SOHO MDI and SDO HMI instruments have diffraction-limited resolutions of 1.25 and 0.91 arcsec (Schou et al. 2011), respectively. To assess the impact of this resolution difference on sunspot area measurements, masks were created for SDO and SOHO images acquired on the same day (less than 6 h apart). Fig. D1 shows an example sunspot group from September 2010 – note the fewer individual sunspots visible in the SOHO image, likely due to either the merging of nearby sunspots or the omission of small sunspots.

![[28.jpg|Figure D1]]
**Figure D1:** SOHO (left) and SDO (right) masks for AR 11109 from 2010 September 26 (top row) and September 29 (bottom row).

Fig. D2 shows a comparison of SOHO and SDO total projected and corrected areas for a selection of 80 d during 2010 and 2011. A SOHO maskThreshold value of 147 was chosen to provide good visual consistency between SOHO and SDO sunspot masks (as illustrated above) and to ensure reliable comparison of the projected areas.

In a few instances, sunspots near the limb were excluded because they merged with false limb mask pixels. The linear fit slopes for projected and corrected areas are 1.004 and 0.987, respectively. For the corrected area plot, the data points that lie away from the linear fit occur when a large sunspot appeared near the limb when a small difference in the number of mask pixels between the SOHO and SDO images has a bigger impact on the corrected area.

Thus, despite the issues with the 2010 and 2011 SOHO images, there is good area agreement between sunspot areas derived from SOHO and SDO data, even though the two instruments have different resolutions.

![[29.jpg|Figure D2]]
**Figure D2:** Comparison of SOHO and SDO total projected (top) and corrected areas (bottom) from 2010 and 2011.