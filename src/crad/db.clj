(ns crad.db 
  (:require [clojure.set :as set]))

(defonce !DB (atom #{}))

(defn reset-db! []
  (reset! !DB #{}))

(defn conj-db! [v]
  (swap! !DB conj v))

(defn in-db [id]
  (some #{id} (map :id @!DB)))

(defn get-by-id [id]
  (first (filter (fn [v] (= id (:id v))) @!DB)))

(defn update-db! [id & {:as opts}]
  (swap! !DB (fn [v]
               (set (map (fn [{did :id
                               :as v}]
                           (if (= id did)
                             (merge v opts)
                             v)) v))))
  (get-by-id id))

(defn remove-items [& ids]
  (set/intersection (set (map :id @!DB)) (set ids)))

(defn keep-items [& ids]
  (let [res (set/intersection (set (map :id @!DB)) (set ids))
        res (set (map get-by-id res))]
    (reset! !DB res)))