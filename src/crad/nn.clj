(ns crad.nn
  (:require [crad.engine :refer [<v>] :as e]
            [crad.util :refer [zip rrand]]))

(defprotocol Module
  (zero-grad [self] "Implement Zero grad")
  (parameters [self] "Define parameters"))

(defrecord NeuronRecord [w b nonlin?]
  clojure.lang.IFn
  (invoke [self xs]
    (let [sxs (map (fn [[wi xi]] (e/v* wi xi)) (zip (:w self) xs))          
          act (reduce (fn [acc v]
                        (e/v+ acc v)) (:b self) sxs)]
      (if (:nonlin? self) (e/relu act) act)))

  Module
  (zero-grad  [self] (let [p (mapv (fn [v] (assoc v :grad 0)) (parameters self))]
                       (->NeuronRecord (drop-last p) (last p) (:nonlin? self))))
  (parameters [self] (conj (:w self) (:b self))))

(defrecord LayerRecord [neurons]
  clojure.lang.IFn
  (invoke [self xs]
    (let [out (mapv (fn [n] (n xs)) (:neurons self))]
      (case (count out)
        1   (first out)
        out)))

  Module
  (zero-grad  [self] (->LayerRecord (mapv zero-grad (:neurons self))))
  (parameters [self] (vec (flatten (map parameters (:neurons self))))))

(defrecord MLPRecord [layers]
  clojure.lang.IFn
  (invoke [self xs]
    (last (mapv (fn [layer] (layer xs)) (:layers self))))

  Module
  (zero-grad  [self] (->MLPRecord (mapv zero-grad (:layers self))))
  (parameters [self] (vec (flatten (mapv parameters (:layers self))))))

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
