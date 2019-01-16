// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <set>
#include <unordered_map>
#include <utility>

#include "utils.hpp"
#include "descriptor.hpp"
#include "distance.hpp"



///
/// \brief The TrackerParams struct stores parameters of PedestrianTracker
///
struct TrackerParams {
    size_t min_track_duration;  ///< Min track duration in milliseconds.

    size_t forget_delay;  ///< Forget about track if the last bounding box in
                          /// track was detected more than specified number of
                          /// frames ago.

    float aff_thr_fast;  ///< Affinity threshold which is used to determine if
                         /// tracklet and detection should be combined (fast
                         /// descriptor is used).

    float aff_thr_strong;  ///< Affinity threshold which is used to determine if
                           /// tracklet and detection should be combined(strong
                           /// descriptor is used).

    float shape_affinity_w;  ///< Shape affinity weight.

    float motion_affinity_w;  ///< Motion affinity weight.

    float time_affinity_w;  ///< Time affinity weight.

    float min_det_conf;  ///< Min confidence of detection.

    cv::Vec2f bbox_aspect_ratios_range;  ///< Bounding box aspect ratios range.

    cv::Vec2f bbox_heights_range;  ///< Bounding box heights range.

    int predict;  ///< How many frames are used to predict bounding box in case
    /// of lost track.

    float strong_affinity_thr;  ///< If 'fast' confidence is greater than this
                                /// threshold then 'strong' Re-ID approach is
                                /// used.

    float reid_thr;  ///< Affinity threshold for re-identification.

    bool drop_forgotten_tracks;  ///< Drop forgotten tracks. If it's enabled it
                                 /// disables an ability to get detection log.

    int max_num_objects_in_track;  ///< The number of objects in track is
                                   /// restricted by this parameter. If it is negative or zero, the max number of
                                   /// objects in track is not restricted.

    ///
    /// Default constructor.
    ///
    TrackerParams();
};

///
/// \brief The Track struct describes tracks.
///
struct Track {
    ///
    /// \brief Track constructor.
    /// \param objs Detected objects sequence.
    /// \param last_image Image of last image in the detected object sequence.
    /// \param descriptor_fast Fast descriptor.
    /// \param descriptor_strong Strong descriptor (reid embedding).
    ///
    Track(const TrackedObjects &objs, const cv::Mat &last_image,
          const cv::Mat &descriptor_fast, const cv::Mat &descriptor_strong)
        : objects(objs),
        predicted_rect(!objs.empty() ? objs.back().rect : cv::Rect()),
        last_image(last_image),
        descriptor_fast(descriptor_fast),
        descriptor_strong(descriptor_strong),
        lost(0),
        length(1) {
            PT_CHECK(!objs.empty());
            first_object = objs[0];
        }

    ///
    /// \brief empty returns if track does not contain objects.
    /// \return true if track does not contain objects.
    ///
    bool empty() const { return objects.empty(); }

    ///
    /// \brief size returns number of detected objects in a track.
    /// \return number of detected objects in a track.
    ///
    size_t size() const { return objects.size(); }

    ///
    /// \brief operator [] return const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return const reference to detected object with specified index.
    ///
    const TrackedObject &operator[](size_t i) const { return objects[i]; }

    ///
    /// \brief operator [] return non-const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return non-const reference to detected object with specified index.
    ///
    TrackedObject &operator[](size_t i) { return objects[i]; }

    ///
    /// \brief back returns const reference to last object in track.
    /// \return const reference to last object in track.
    ///
    const TrackedObject &back() const {
        PT_CHECK(!empty());
        return objects.back();
    }

    ///
    /// \brief back returns non-const reference to last object in track.
    /// \return non-const reference to last object in track.
    ///
    TrackedObject &back() {
        PT_CHECK(!empty());
        return objects.back();
    }

    TrackedObjects objects;   ///< Detected objects;
    cv::Rect predicted_rect;  ///< Rectangle that represents predicted position
                              /// and size of bounding box if track has been lost.
    cv::Mat last_image;       ///< Image of last detected object in track.
    cv::Mat descriptor_fast;  ///< Fast descriptor.
    cv::Mat descriptor_strong;  ///< Strong descriptor (reid embedding).
    size_t lost;                ///< How many frames ago track has been lost.

    TrackedObject first_object;  ///< First object in track.
    size_t length;  ///< Length of a track including number of objects that were
                    /// removed from track in order to avoid memory usage growth.
};

///
/// \brief Online pedestrian tracker algorithm implementation.
///
/// This class is implementation of pedestrian tracking system. It uses two
/// different appearance measures to compute affinity between bounding boxes:
/// some fast descriptor and some strong descriptor. Each time the assignment
/// problem is solved. The assignment problem in our case is how to establish
/// correspondence between existing tracklets and recently detected objects.
/// First step is to compute an affinity matrix between tracklets and
/// detections. The affinity equals to
///       appearance_affinity * motion_affinity * shape_affinity.
/// Where appearance is 1 - distance(tracklet_fast_dscr, detection_fast_dscr).
/// Second step is to solve the assignment problem using Kuhn-Munkres
/// algorithm. If correspondence between some tracklet and detection is
/// established with low confidence (affinity) then the strong descriptor is
/// used to determine if there is correspondence between tracklet and detection.
///
class PedestrianTracker {
public:
    using Descriptor = std::shared_ptr<IImageDescriptor>;
    using Distance = std::shared_ptr<IDescriptorDistance>;

    ///
    /// \brief Constructor that creates an instance of the pedestrian tracker with
    /// parameters.
    /// \param[in] params - the pedestrian tracker parameters.
    ///
    explicit PedestrianTracker(const TrackerParams &params = TrackerParams());
    virtual ~PedestrianTracker() {}

    ///
    /// \brief Process given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \param[in] detections Detected objects on the frame.
    /// \param[in] timestamp Timestamp must be positive and measured in
    /// milliseconds
    ///
    void Process(const cv::Mat &frame, const TrackedObjects &detections,
                 uint64_t timestamp);

    ///
    /// \brief Pipeline parameters getter.
    /// \return Parameters of pipeline.
    ///
    const TrackerParams &params() const;

    ///
    /// \brief Pipeline parameters setter.
    /// \param[in] params Parameters of pipeline.
    ///
    void set_params(const TrackerParams &params);

    ///
    /// \brief Fast descriptor getter.
    /// \return Fast descriptor used in pipeline.
    ///
    const Descriptor &descriptor_fast() const;

    ///
    /// \brief Fast descriptor setter.
    /// \param[in] val Fast descriptor used in pipeline.
    ///
    void set_descriptor_fast(const Descriptor &val);

    ///
    /// \brief Strong descriptor getter.
    /// \return Strong descriptor used in pipeline.
    ///
    const Descriptor &descriptor_strong() const;

    ///
    /// \brief Strong descriptor setter.
    /// \param[in] val Strong descriptor used in pipeline.
    ///
    void set_descriptor_strong(const Descriptor &val);

    ///
    /// \brief Fast distance getter.
    /// \return Fast distance used in pipeline.
    ///
    const Distance &distance_fast() const;

    ///
    /// \brief Fast distance setter.
    /// \param[in] val Fast distance used in pipeline.
    ///
    void set_distance_fast(const Distance &val);

    ///
    /// \brief Strong distance getter.
    /// \return Strong distance used in pipeline.
    ///
    const Distance &distance_strong() const;

    ///
    /// \brief Strong distance setter.
    /// \param[in] val Strong distance used in pipeline.
    ///
    void set_distance_strong(const Distance &val);

    ///
    /// \brief Returns number of counted people.
    /// \return a number of counted people.
    ///
    size_t Count() const;

    ///
    /// \brief Returns a detection log which is used for tracks saving.
    /// \param[in] valid_only If it is true the method returns valid track only.
    /// \return a detection log which is used for tracks saving.
    ///
    DetectionLog GetDetectionLog(const bool valid_only) const;

    ///
    /// \brief Get active tracks to draw
    /// \return Active tracks.
    ///
    std::unordered_map<size_t, std::vector<cv::Point> > GetActiveTracks() const;

    ///
    /// \brief Get tracked detections.
    /// \return Tracked detections.
    ///
    TrackedObjects TrackedDetections() const;

    ///
    /// \brief Draws active tracks on a given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \return Colored image with drawn active tracks.
    ///
    cv::Mat DrawActiveTracks(const cv::Mat &frame);

    ///
    /// \brief Print confusion matrices of data association classifiers.
    /// It works only in case of loaded detection logs instead of native
    /// detectors.
    ///
    void PrintConfusionMatrices() const;

    ///
    /// \brief IsTrackForgotten returns true if track is forgotten.
    /// \param id Track ID.
    /// \return true if track is forgotten.
    ///
    bool IsTrackForgotten(size_t id) const;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Set of tracks {id, track}.
    ///
    const std::unordered_map<size_t, Track> &tracks() const;

    ///
    /// \brief IsTrackValid Checks whether track is valid (duration > threshold).
    /// \param track_id Index of checked track.
    /// \return True if track duration exceeds some predefined value.
    ///
    bool IsTrackValid(size_t track_id) const;

    ///
    /// \brief DropForgottenTracks Removes tracks from memory that were lost too
    /// many frames ago.
    ///
    void DropForgottenTracks();

    ///
    /// \brief DropForgottenTracks Check that the track was lost too many frames
    /// ago
    /// and removes it frm memory.
    ///
    void DropForgottenTrack(size_t track_id);

    ///
    /// \brief Prints reid performance counter
    ///
    void PrintReidPerformanceCounts() const;

private:
    struct Match {
        int frame_idx1;
        int frame_idx2;
        cv::Rect rect1;
        cv::Rect rect2;
        cv::Rect pr_rect1;
        bool pr_label;
        bool gt_label;

        Match() {}

        Match(const TrackedObject &a, const cv::Rect &a_pr_rect,
              const TrackedObject &b, bool pr_label)
            : frame_idx1(a.frame_idx),
            frame_idx2(b.frame_idx),
            rect1(a.rect),
            rect2(b.rect),
            pr_rect1(a_pr_rect),
            pr_label(pr_label),
            gt_label(a.object_id == b.object_id) {
                PT_CHECK_NE(frame_idx1, frame_idx2);
            }
    };


    const ObjectTracks all_tracks(bool valid_only) const;
    // Returns shape affinity.
    static float ShapeAffinity(float w, const cv::Rect &trk, const cv::Rect &det);

    // Returns motion affinity.
    static float MotionAffinity(float w, const cv::Rect &trk,
                                const cv::Rect &det);

    // Returns time affinity.
    static float TimeAffinity(float w, const float &trk, const float &det);

    cv::Rect PredictRect(size_t id, size_t k, size_t s) const;

    cv::Rect PredictRectSmoothed(size_t id, size_t k, size_t s) const;

    cv::Rect PredictRectSimple(size_t id, size_t k, size_t s) const;

    void SolveAssignmentProblem(
        const std::set<size_t> &track_ids, const TrackedObjects &detections,
        const std::vector<cv::Mat> &descriptors, float thr,
        std::set<size_t> *unmatched_tracks,
        std::set<size_t> *unmatched_detections,
        std::set<std::tuple<size_t, size_t, float>> *matches);

    void ComputeFastDesciptors(const cv::Mat &frame,
                               const TrackedObjects &detections,
                               std::vector<cv::Mat> *desriptors);

    void ComputeDissimilarityMatrix(const std::set<size_t> &active_track_ids,
                                    const TrackedObjects &detections,
                                    const std::vector<cv::Mat> &fast_descriptors,
                                    cv::Mat *dissimilarity_matrix);

    std::vector<float> ComputeDistances(
        const cv::Mat &frame,
        const TrackedObjects& detections,
        const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
        std::map<size_t, cv::Mat> *det_id_to_descriptor);

    std::map<size_t, std::pair<bool, cv::Mat>> StrongMatching(
        const cv::Mat &frame,
        const TrackedObjects& detections,
        const std::vector<std::pair<size_t, size_t>> &track_and_det_ids);

    std::vector<std::pair<size_t, size_t>> GetTrackToDetectionIds(
        const std::set<std::tuple<size_t, size_t, float>> &matches);

    float AffinityFast(const cv::Mat &descriptor1, const TrackedObject &obj1,
                       const cv::Mat &descriptor2, const TrackedObject &obj2);

    float Affinity(const TrackedObject &obj1, const TrackedObject &obj2);

    void AddNewTrack(const cv::Mat &frame, const TrackedObject &detection,
                     const cv::Mat &fast_descriptor,
                     const cv::Mat &descriptor_strong = cv::Mat());

    void AddNewTracks(const cv::Mat &frame, const TrackedObjects &detections,
                      const std::vector<cv::Mat> &descriptors_fast);

    void AddNewTracks(const cv::Mat &frame, const TrackedObjects &detections,
                      const std::vector<cv::Mat> &descriptors_fast,
                      const std::set<size_t> &ids);

    void AppendToTrack(const cv::Mat &frame, size_t track_id,
                       const TrackedObject &detection,
                       const cv::Mat &descriptor_fast,
                       const cv::Mat &descriptor_strong);

    bool EraseTrackIfBBoxIsOutOfFrame(size_t track_id);

    bool EraseTrackIfItWasLostTooManyFramesAgo(size_t track_id);

    bool UpdateLostTrackAndEraseIfItsNeeded(size_t track_id);

    void UpdateLostTracks(const std::set<size_t> &track_ids);

    static cv::Mat ConfusionMatrix(const std::vector<Match> &matches);

    const std::set<size_t> &active_track_ids() const;

    // Returns decisions made by heuristic based on fast distance/descriptor and
    // shape, motion and time affinity.
    const std::vector<Match> & base_classifier_matches() const;

    // Returns decisions made by heuristic based on strong distance/descriptor
    // and
    // shape, motion and time affinity.
    const std::vector<Match> &reid_based_classifier_matches() const;

    // Returns decisions made by strong distance/descriptor affinity.
    const std::vector<Match> &reid_classifier_matches() const;

    TrackedObjects FilterDetections(const TrackedObjects &detections) const;
    bool IsTrackForgotten(const Track &track) const;

    // Parameters of the pipeline.
    TrackerParams params_;

    // Indexes of active tracks.
    std::set<size_t> active_track_ids_;

    // Descriptor fast (base classifer).
    Descriptor descriptor_fast_;

    // Distance fast (base classifer).
    Distance distance_fast_;

    // Descriptor strong (reid classifier).
    Descriptor descriptor_strong_;

    // Distance strong (reid classifier).
    Distance distance_strong_;

    // All tracks.
    std::unordered_map<size_t, Track> tracks_;

    // Previous frame image.
    cv::Size prev_frame_size_;

    struct pair_hash {
        std::size_t operator()(const std::pair<size_t, size_t> &p) const {
            PT_CHECK(p.first < 1e6 && p.second < 1e6);
            return p.first * 1e6 + p.second;
        }
    };

    // Distance between current active tracks.
    std::unordered_map<std::pair<size_t, size_t>, float, pair_hash> tracks_dists_;

    // Whether collect matches and compute confusion matrices for
    // track-detection
    // association task (base classifier, reid-based classifier,
    // reid-classiifer).
    bool collect_matches_;

    // This vector contains decisions made by
    // fast_apperance-motion-shape affinity model.
    std::vector<Match> base_classifier_matches_;

    // This vector contains decisions made by
    // strong_apperance(cnn-reid)-motion-shape affinity model.
    std::vector<Match> reid_based_classifier_matches_;

    // This vector contains decisions made by
    // strong_apperance(cnn-reid) affinity model only.
    std::vector<Match> reid_classifier_matches_;

    // Number of all current tracks.
    size_t tracks_counter_;

    // Number of dropped valid tracks.
    size_t valid_tracks_counter_;

    cv::Size frame_size_;

    std::vector<cv::Scalar> colors_;

    uint64_t prev_timestamp_;
};

