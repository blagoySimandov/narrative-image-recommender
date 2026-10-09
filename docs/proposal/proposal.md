# Narrative Photo Selection and Sequencing

**Project Proposal**

## 1. Problem Statement

Many users have large collections of unordered digital photographs.
Existing tools summarize photos with timestamps or text captions.
But real photo albums do not always follow chronological order.

This project develops an automated system that selects N photos from a
collection of K photos, orders them and aims for maximum coverage and coherent order.
The system must satisfy two goals:

1. Select photos that have high visual quality and broad coverage of the wider dataset.
2. Arrange the selected photos in a coherent order that makes sense to a human viewer.

E.g. a photo of a beach followed by a photo of a snowy mountain into another photo of a beach would be a bad sequence.

## 2. Implementation Plan

1. Explore, Find and Collect appropriate datasets for the problem.
   - The dataset should be large enough to have a good evaluation.
   - The dataset should have a wide range of photos.
   - The dataset should have photos with high visual quality as well as ones with low visual quality.
   - It should be separated by subjects - e.g. user1 went on a trip to japan and has 150 photos, user2 has 200 photos from
     a birthday party....
   - Ideally it should be labeled in some way - this is not a requirement.

2. Develop a system that can select N photos from the dataset.
   - The system should be able to select photos that are not in chronological order.
   - The system should be able to select photos that are in chronological order.
   - The system should be "tunable" in multiple ways so that selection can be matched to user prefences.

3. Evaluate the system.
   - Evaluate the system on the dataset. (if labeled)
   - If needed survey users to evaluate the system.

4. Write a report outlining the results and discoveries.
