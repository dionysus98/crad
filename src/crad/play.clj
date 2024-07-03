(ns crad.play
  (:require [clojure.pprint :as pprint]
            [crad.engine :refer [<v>] :as e]
            [crad.nn :as nn]
            [crad.util :as util]))

(def data
  #_(update (util/make-moons) :y
            (partial mapv (fn [v] (- (Math/pow v 2) 1))))
  {:X [[2.0 3.0 -1.0]
       [3.0 -1.0 0.5]
       [0.5 1.0 1.0]
       [1.0 1.0 -1.0]]
   :y [1.0 -1.0 -1.0 1.0]})

(def model (nn/<MLP> 2 [4 4 1]))

(defn loss
  [model {xb :X
          yb :y}]
  (let [one        (<v> 1.0)
        sum        (partial reduce e/v+)
        inputs     (mapv (fn [xrow] (mapv (fn [x] (<v> x)) xrow)) xb)
        res        (reduce (fn [m i]
                             (let [res (m i)]
                               (-> (update m :output conj (:output res))
                                   (assoc :model (:mlp res))))) {:otput []
                                                                 :model model}
                           inputs)
        preds      (:output res)
        model      (:model res)
        ufn        (fn [[yi, scorei]]
                     (e/relu (e/v+ one (e/v* (<v> (- yi)) scorei))))
        pfn        (fn [ps] (util/zip yb ps))
        ypreds     (nn/nmap
                    model {:ufn ufn
                           :pfn pfn})
        alpha      (<v> 1e-4)
        olayer     (-> ypreds :layers last)
        ps         (nn/parameters olayer)
        data-loss  (e/v* (sum ps) (e/vdiv one (<v> (count ps))))
        reg-loss   (e/v* alpha (sum (map (fn [p] (e/v* p p)) ps)))
        total-loss (e/v+ data-loss reg-loss)
        accuracy   (mapv (fn [[yi si]]
                           (if (and  (> yi 0) (> (:data si) 0))
                             1.0 0.0))
                         (util/zip yb (nn/parameters olayer)))]
    {:total-loss total-loss
     :loss-model ypreds
     :accuracy   (/ (apply + accuracy) (count accuracy))}))

(def !model (atom nil))

(defn optz
  [model data n]
  (reset! !model model)
  (let [loss (loss model data)
        tl   (-> loss
                 :loss-model
                 nn/zero-grad
                 nn/backward)
        tld  (-> tl :layers last :neurons last :b :data)]
    (if (> n 0)
      (let [;; learning-rate
            lr (- 1.0 (* 0.9 (/ n 100)))
            um (nn/nmap
                tl
                (fn [p] (update p :data #(- % (* lr (:grad p))))))]
        (pprint/pprint {:step       n
                        :loss-model tld
                        :accuracy   (:accuracy loss)})
        (optz um data (dec n)))
      {:loss-model tld
       :accuracy   (:accuracy loss)})))

(try
  (optz model data 50)
  (catch Exception e e))

(first (:w (last (:neurons (first (:layers (model (mapv <v> [8 9 3 4 9]))))))))
(first (:w (last (:neurons (first (:layers (@!model (mapv <v> [8 9 3 4 9]))))))))

@!model

(first (:neurons (first (:layers (model (mapv <v> [8 9 3 4 9]))))))
(first (:neurons (first (:layers (nn/backward (model (mapv <v> [8 9 3 4 9])))))))

(update [1 2 3] 2 inc)

((nn/<MLP> 2 [2 2 1]) [0.2 0.3 0.4 0.5])

(mapv (comp count :neurons) (:layers ((nn/<MLP> 2 [2 2 1]) [0.2 0.3])))

(def !model- (atom (nn/<MLP> 2 [2 2 1])))

@!model-

(try
 ((nn/<MLP> 2 [2 2 1]) [0.2 0.3 0.4 0.6])
  (catch Exception e e))

(nn/backward (nn/<MLP> 2 [2 2 4 1]))

(count (:layers (nn/<MLP> 2 [2 2 4 1])))

(((nn/<MLP> 2 [2 2 1]) [1]) [2])

(def !mine (atom (nn/<MLP> 2 [2 2 1])))

(try
  (let [model      @!mine
        xb         (:X data)
        yb         (:y data)
        one        (<v> 1.0)
        sum        (partial reduce e/v+)
        model      (reduce (fn [model item] (model item)) model xb)
        ufn        (fn [[yi, scorei]]
                     (<v> (:data (e/relu (e/v+ one (e/v* (<v> (- yi)) scorei))))))
        pfn        (fn [ps] (util/zip yb ps))
        layerc     (count (:layers model))
        model      (reduce (fn [model layer]
                             (update-in model [:layers layer]
                                        (fn [v]
                                          (nn/nmap
                                           v {:ufn ufn
                                              :pfn pfn}))))
                           model (reverse (range layerc)))
        alpha      (<v> 1e-4)        
        accuracy   (mapv (fn [[yi si]]
                           (if (and  (> yi 0) (> (:data si) 0))
                             1.0 0.0))
                         (util/zip yb (-> model :layers last nn/parameters)))
        ;; layerc     (count (:layers model))
        ;; model      (update-in model [:layers layerc :n])
        ]
    (reset! !mine (nn/zero-grad model))
    ;; (e/backward total-loss)
    model)
  (catch Exception e e))

((nn/<MLP> 2 [2 2 1]) [7 8 9])

(def mev (nn/<MLP> 2 [2 2 1]))

mev
