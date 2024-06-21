(ns crad.engine)

(defprotocol ValueProtocol
  (backward [self] "implements the back propagation for the Value")
  (v+ [self other] "implements addition")
  (v- [self other] "implements subtraction")
  (v* [self other] "implements multiplication")
  (vdiv [self other] "implements division")
  (v** [self pow] "implements pow")
  (relu [self] "implements RELU"))

(defrecord ValueRecord [data grad backward children op]
  ValueProtocol
  (backward [self]
    (let [>bk #((:backward %) %)
          bk- (fn backward- [s]
                (let [res (reduce
                           (fn build [acc v]
                             (if (some #{v} acc)
                               acc
                               (concat
                                (backward- (or (not-empty (>bk v)) v))
                                acc)))
                           []
                           (:children s))]
                  (vec (cons s res))))
          out (>bk (assoc self :grad 1))]
      (vec (remove nil? (bk- out)))))

  (relu [self]
    (let [n  (:data self)
          bk (fn [out]
               (update-in out [:children 0 :grad]
                          (partial + (if (> (:data out) 0)
                                       (:grad out)
                                       0))))]
      (->ValueRecord (if (< n 0) 0 n)
                     (:grad self)
                     bk
                     [self]
                     'relu)))

  (v+ [self other]
    (let [bk (fn [out]
               (let [o (-> out
                           (update-in [:children 0 :grad] (partial + (:grad out)))
                           (update-in [:children 1 :grad] (partial + (:grad out))))]
                 o))]
      (->ValueRecord (+ (:data self) (:data other))
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
      (->ValueRecord (* (:data self) (:data other))
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
      (->ValueRecord (Math/pow (:data self) pow)
                     (:grad self)
                     bk
                     [self]
                     (symbol (str "**" pow))))))

(defmethod print-method ValueRecord [v ^java.io.Writer w]
  (.write w (pr-str (-> (dissoc v :backward)
                        (update :children count)
                        (update-vals (fn [v]
                                       (if (number? v)
                                         (parse-double (format "%.3f" (double v)))
                                         v)))))))

(defn value
  [n & {grad :grad
        cn :children
        op :op}]
  (let [grad (or grad 0)
        cn   (or cn [])
        op   (or op nil)
        bk   (constantly nil)]
    (->ValueRecord n grad bk cn op)))



(comment

  (let [a (value 2)
        b (value 4)
        c (value 5)
        d (v+ a (v+ b c))
        res (relu d)]
    (backward res))


  :rcf)

