(ns crad.play
  (:require [crad.engine :refer [<v>] :as e]
            [crad.nn :as nn]
            [crad.util :as util]))

(def data
  (update (util/make-moons) :y
          (partial mapv (fn [v] (- (Math/pow v 2) 1)))))

(def model (nn/<MLP> 2 [5 5 1]))

(defn loss
  [model {xb :X
          yb :y}]
  (let [zero       (<v> 0.0)
        one        (<v> 1.0)
        sum        (partial reduce e/v+ zero)
        inputs     (mapv (fn [xrow] (mapv (fn [x] (<v> x)) xrow)) xb)
        scores     (mapv model inputs)
        yscores    (util/zip yb scores)
        losses     (mapv (fn [[yi, scorei]]
                           (e/relu
                            (e/v+ one
                                  (e/v* (e/v* (<v> -1.0)
                                              (<v> yi))
                                        scorei))))
                         yscores)
        data-loss  (e/v* (sum losses)
                         (e/vdiv one (<v> (count losses))))
        alpha      (<v> 1e-4)
        reg-loss   (e/v* alpha (sum (map (fn [p] (e/v* p p)) (nn/parameters model))))
        total-loss (e/v+ data-loss reg-loss)
        accuracy   (mapv (fn [[yi si]]
                           (if (and  (> yi 0) (> (:data si) 0))
                             1.0 0.0))
                         yscores)]
    {:total-loss total-loss
     :accuracy (/ (apply + accuracy) (count accuracy))}))

(update (loss model data) :total-loss e/backward)

{:total-loss 10613.915778705134
 :accuracy   0.0}

(nn/zero-grad model)



