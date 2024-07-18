(ns crad.play
  (:require [crad.engine :refer [<v>] :as e]
            [crad.nn :as nn]
            [crad.util :as util]
            [crad.db :as db]))

(def model (nn/<MLP> 3 [4 4 1]))

(def !xs (mapv (partial mapv <v>)
               [[2.0 3.0 -1.0]
                [3.0 -1.0 0.5]
                [0.5 1.0 1.0]
                [1.0 1.0 -1.0]]))

(def !ys (mapv <v> [1.0 -1.0 -1.0 1.0]))

(defn loss [k]
  (let [ypred (for [x !xs] (nn/feed-forward model x))
        cost  (reduce e/v+
                      (for [[ygt yout] (util/zip !ys ypred)]
                        (e/v** (e/v- (db/get-by-id (:id yout))
                                     (db/get-by-id (:id ygt)))
                               2.0)))]
   ;;  zero-grad
    (for [p    (nn/parameters model)
          :let [p (db/get-by-id (:id p))]]
      (db/update-db! (:id p) :grad 0.0))
   ;;  backward
    (e/backward cost)
   ;; step
    (for [p    (nn/parameters model)
          :let [p (db/get-by-id (:id p))]]
      (db/update-db! (:id p) :data (+ (:data p) ((- 0.01) * (:grad p)))))
    ;; clear cache
    (apply db/keep-items
           (concat
            (map :id (nn/parameters model))
            (map  :id (flatten !xs))
            (map :id !ys)))
    cost))

(try
  (doseq  [k (range 50)]
    (println (loss k)))
  (catch Exception e e))

(for [p (nn/parameters model)]
  (:id p))

(count @db/!DB)

(nn/parameters model)

(db/get-by-id  :VSGHDFUDI)
