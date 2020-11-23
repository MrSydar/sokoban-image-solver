(define 

	(domain sokoban)
	
	(:requirements :adl)

	(:predicates
		(box ?x)
		(player ?x)
		(horizontal ?x ?y)
		(vertical ?x ?y)
	)
	
	(:action go
		:parameters (?from ?to)
		:precondition
		(and
			(player ?from)
			(or
				(horizontal ?from ?to)
				(vertical ?from ?to)
				(horizontal ?to ?from)
				(vertical ?to ?from)
			)
			(not (box ?to))
		)
		:effect
		(and 
			(player ?to)
			(not (player ?from))
		)
	)

	(:action push
		:parameters (?from ?to ?topaczka)
		:precondition
		(and
			(player ?from)
			(box ?to)
			(not (box ?topaczka))
			(not (= ?from ?to))
			(not (= ?from ?topaczka))
			(not (= ?to ?topaczka))
			(or
				(and
					(or 
						(horizontal ?from ?to)
						(horizontal ?to ?from)
					)
					(or
						(horizontal ?to ?topaczka)
						(horizontal ?topaczka ?to)
					)
					(not
						(or
							(horizontal ?from ?topaczka)
							(horizontal ?topaczka ?from)
						)
					)
				)
				(and
					(or 
						(vertical ?from ?to)
						(vertical ?to ?from)
					)
					(or
						(vertical ?to ?topaczka)
						(vertical ?topaczka ?to)
					)
					(not
						(or
							(vertical ?from ?topaczka)
							(vertical ?topaczka ?from)
						)
					)
				)
			)
		)
		:effect
		(and 
			(player ?to)
			(not (player ?from))
			(box ?topaczka)
			(not (box ?to))
		)
	)
)
