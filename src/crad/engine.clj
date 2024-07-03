(ns crad.engine
  (:require [crad.util :as util]))

(defn rid []
  (->> (repeatedly 50 #(char (int (util/rrand 65 90))))
       (apply str)
       keyword))

(defprotocol ValueProtocol
  (parameters [self] "Lists out the params/children in a value")
  (outputs [self] "Lists out the params/children in a value")
  (inputs [self] "Lists out the params/children in a value")
  (backward [self] "implements back propagation")
  (v+ [self other] "implements addition")
  (v- [self other] "implements subtraction")
  (v* [self other] "implements multiplication")
  (vdiv [self other] "implements division")
  (v** [self pow] "implements pow")
  (relu [self] "implements RELU"))

(defrecord ValueRecord [id data grad backward children op]
  ValueProtocol
  (parameters [self]
    (let [ps-  (fn params-
                 ([xs s] (if (some #{s} xs)
                           xs
                           (let [cxs (conj xs (dissoc s :children))]
                             (if-let [cl (not-empty (:children s))]
                               (reduce params- cxs cl)
                               (vec (distinct cxs))))))
                 ([s] (params- [] s)))
          ps (reverse (ps- self))
          vr (fn [v]
               (->ValueRecord (:id v)
                              (:data v)
                              (:grad v)
                              (:backward v)
                              (:children v)
                              (:op v)))]
      (mapv vr ps)))

  (inputs [self]
    (filterv (comp not :op) (parameters self)))

  (outputs [self]
    (filterv :op (parameters self)))

  (backward [self]
    (let [bk-  (fn backward-
                 [s & {grad :grad}]
                 (let [v (if (number? grad)
                           (assoc s :grad grad)
                           s)
                       >bk  #((:backward %) %)]
                   (-> (or (not-empty (>bk v)) v)
                       (update :children (partial mapv backward-)))))]
      (bk- self :grad 1)))

  (relu [self]
    (let [n    (:data self)
          bk   (fn [out]
                 (update-in out [:children 0 :grad]
                            (partial + (if (> (:data out) 0)
                                         (:grad out)
                                         0))))]
      (->ValueRecord (rid)
                     (if (< n 0) 0 n)
                     (:grad self)
                     bk
                     [self]
                     'relu)))

  (v+ [self other]
    (let [bk    (fn [out]
                  (let [o (-> out
                              (update-in [:children 0 :grad] (partial + (:grad out)))
                              (update-in [:children 1 :grad] (partial + (:grad out))))]
                    o))]
      (->ValueRecord (rid)
                     (+ (:data self) (:data other))
                     (:grad self)
                     bk
                     [self, other]
                     '+)))

  (v- [self other]
    (-> (v+ self (update other :data (partial * -1)))
        (assoc :op '-)))

  (v* [self other]
    (let [bk (fn [out]
               (-> out
                   (update-in [:children 0 :grad] (partial + (* (:data other) (:grad out))))
                   (update-in [:children 1 :grad] (partial + (* (:data self) (:grad out))))))]
      (->ValueRecord (rid)
                     (* (:data self) (:data other))
                     (:grad self)
                     bk
                     [self, other]
                     '*)))

  (vdiv [self other] (-> (v* self (v** other -1))
                         (assoc :op 'div)))

  (v** [self pow]
    (let [bk  (fn [out]
                (update-in out [:children 0 :grad]
                           (partial + (* (:grad out)
                                         pow
                                         (Math/pow (:data self) (dec pow))))))]
      (->ValueRecord (rid)
                     (Math/pow (:data self) pow)
                     (:grad self)
                     bk
                     [self]
                     (symbol (str "**" pow))))))

(defmethod print-method ValueRecord [v ^java.io.Writer w]
  (.write w (pr-str (-> (dissoc v :backward :id)
                        (update-vals (fn [v]
                                       (if (number? v)
                                         (parse-double (format "%.3f" (double v)))
                                         v)))))))

(defn <v>
  [n & {id   :id
        grad :grad
        cn   :children
        op   :op}]
  (let [id   (or id (rid))
        grad (or grad 0)
        cn   (or cn [])
        op   (or op nil)
        bk   (constantly nil)]
    (->ValueRecord id n grad bk cn op)))

(comment
  (let [a   (<v> 2)
        b   (<v> 4)
        c   (<v> 5)
        e   (<v> 1)
        d   (->> c
                 (v+ b)
                 (v+ a)
                ;;  (v* e)
                 )
        res (relu (v+ c (v+ a b)))]
    ;; (inputs (backward (backward (backward res))))
    ;; (count (parameters res))
    ;; (inputs res)
    [(inputs res)
     (inputs (backward (backward (backward res))))]
    ;; (parameters (backward res))
    )
  :rcf)

