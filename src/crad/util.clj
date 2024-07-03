(ns crad.util)

(defn zip
  "Just like Python's `zip`"
  [& colls]
  (let [min  (apply min (map count colls))
        base (repeatedly min vector)
        zfn  (fn zipper [acc n]
               (let [xs (vec (remove nil?
                                     (map #(nth (take min %) n nil)
                                          colls)))]
                 (vec (remove empty? (conj acc xs)))))]
    (reduce zfn base (range min))))

(defn rrand
  [from to
   & {dtype :dtype
      :or   {dtype :float}}]
  (let [rfn  (case dtype
               :float rand
               :int   rand-int)
        rnd  (rfn to)
        rneg (case (rand-int 2)
               0 -1
               1 1)
        rnd (if (neg? from)
              (* rneg rnd)
              rnd)]
    (if (< rnd from)
      (rrand from to :dtype dtype)
      rnd)))

(defn linspace
  "(linspace- -10 15 5.0)"
  [start stop num]
  (let [div   (dec num)
        delta (- stop start)
        step  (/ delta div)]
    (map #(+ start (* % step)) (range num))))

;; (linspace- -10 15 5.0)

(defn make-moons
  "(make-moons)"
  [& {nsa :nsamples
      :or {nsa 100}}]
  (let [nsaout (Math/floorDiv nsa 2)
        nsain  (- nsa nsaout)
        ls     (partial linspace 0 Math/PI)
        ox     (mapv #(Math/cos %) (ls nsaout))
        oy     (mapv #(Math/sin %) (ls nsaout))
        ix     (mapv #(- 1 (Math/cos %)) (ls nsain))
        iy     (mapv #(- (- 1 (Math/sin %)) 0.5) (ls nsain))]
    {:X (zip (shuffle (concat ox ix))
             (shuffle (concat oy iy)))
     :y (concat (shuffle (repeatedly nsaout (constantly 0)))
                (shuffle (repeatedly nsain (constantly 1))))}))

(comment
  (count (:X (make-moons)))
  :rcf)