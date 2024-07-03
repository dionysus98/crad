(ns crad.nn
  (:require [crad.engine :refer [<v>] :as e]
            [crad.util :refer [rrand zip]]
            [crad.nn :as nn]))

(defprotocol Module
  (backward [self] "Define parameters")
  (parameters [self] "Define parameters")
  (nmap [self ufn] "update parameters")
  (zero-grad [self] "Implement Zero grad")
  (sum [self] "Define parameters")
  (loss [self] "Define parameters"))

(defrecord NeuronRecord [w b nonlin?]
  clojure.lang.IFn
  (invoke           [self xs]
    (let [sxs (map (fn [[wi xi]]
                     (e/v* wi (if (number? xi) (<v> xi) xi)))
                   (zip (:w self) xs)) ;; cross product
          act (e/v+ (reduce e/v+ (first sxs) (rest sxs)) (:b self)) ;; add the bias
          act (if (:nonlin? self) (e/relu act) act) ;; activation
          ps  (->> act
                   e/inputs
                   (map #(dissoc % :op))
                   distinct
                   (map (comp <v> :data)))]
      (->NeuronRecord (drop-last ps) (last ps) (:nonlin? self))))

  Module
  (sum [self] (reduce e/v+ (parameters self)))
  (parameters       [self] (conj (:w self) (:b self)))

  (nmap [self ufn]
    (let [ps       (parameters self)
          [ufn ps] (cond
                     (fn? ufn)  [ufn ps]
                     (map? ufn) (do
                                  (assert (and (fn? (:pfn ufn))
                                               (fn? (or (:ufn ufn) identity)))
                                          "Invalid arguements passed to nmaps for NeuronRecord")
                                  (let [pfn (:pfn ufn)
                                        ufn (or (:ufn ufn) identity)]
                                    [ufn (pfn ps)])))
          ps       (mapv ufn ps)]
      (->NeuronRecord (drop-last ps) (last ps) (:nonlin? self))))

  (zero-grad        [self]
    (nmap self (fn [v] (assoc v :grad 0))))

  (backward         [self]
    (nmap self e/backward)))

(defrecord LayerRecord [neurons]
  clojure.lang.IFn
  (invoke           [self xs]
    (->LayerRecord (mapv (fn [n] (n xs)) (:neurons self))))


  Module
  (sum [self] (mapv sum (:neurons self)))
  (parameters       [self]
    (vec (flatten (mapv parameters (:neurons self)))))

  (nmap [self ufn]
    (->LayerRecord (mapv #(nmap % ufn) (:neurons self))))

  (zero-grad        [self]
    (->LayerRecord (mapv zero-grad (:neurons self))))

  (backward        [self]
    (->LayerRecord (mapv backward (:neurons self)))))

(defrecord MLPRecord [layers]

  clojure.lang.IFn
  (invoke [self xs]
    (->MLPRecord (mapv (fn [l] (l xs)) (:layers self))))

  Module
  (loss [self] (count self))
  (sum [self] (mapv sum (:layers self)))
  (parameters       [self]
    (vec (flatten (map parameters (:layers self)))))
  (nmap [self ufn]
    (->MLPRecord (mapv #(nmap % ufn) (:layers self))))
  (zero-grad        [self]
    (->MLPRecord (mapv zero-grad (:layers self))))
  (backward         [self]
    (->MLPRecord (mapv backward (:layers self)))))

(defn <neuron>
  [nin & {nonlin? :nonlin?
          :or     {nonlin? true}}]
  (let [w (mapv (fn [_] (<v> (rrand -1 1)))  (range nin))
        b (<v> 0)]
    (->NeuronRecord w b nonlin?)))

(defn <layer>
  [nin nout & {nonlin? :nonlin?
               :or     {nonlin? true}}]
  (->LayerRecord (mapv (fn [_] (<neuron> nin :nonlin? nonlin?)) (range nout))))

(defn <MLP> [nin nouts]
  (let [sz (cons nin nouts)
        couts (count nouts)
        layer (fn [i]
                (<layer> (nth sz i) (nth sz (inc i))
                         {:nonlin? (not= i couts)}))]
    (->MLPRecord (mapv layer (range couts)))))

(def <n> <neuron>)
(def <l> <layer>)
(def <m> <MLP>)
