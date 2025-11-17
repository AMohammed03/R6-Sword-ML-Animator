--!strict

local module = {}

local HTTPService = game:GetService("HttpService")



local JOINT_NAMES = {
	["Torso"] = true,
	["Right Arm"] = true,
	["Left Arm"] = true, 
	["Head"] = true,
	["Handle"] = true
}


local function poseDataToObject(poseName, poseData, parent: Instance)
	local newPose = Instance.new("Pose")
	newPose.Name = poseName

	local pos: {number}, rot: {number} = poseData.Position, poseData.Rotation
	newPose.CFrame = CFrame.new(pos[1], pos[2], pos[3]) * CFrame.Angles(rot[1], rot[2], rot[3])

	for propName, propData in pairs(poseData) do
		if JOINT_NAMES[propName] then
			poseDataToObject(propName, propData, newPose)
		end

		newPose.Parent = parent
	end

	return newPose
end

function module.ImportAnims(JSONdata, parent)
	assert(JSONdata and type(JSONdata) == "string", "Must pass json data string of animations.")
	
	if not parent then
		parent = workspace
	end
	
	local data = HTTPService:JSONDecode(JSONdata)
	local keyframes = data.keyframes

	local newKFSeq = Instance.new("KeyframeSequence")
	newKFSeq.Name = "GeneratedAnimation_"..tostring(math.random(1, 1000))
	newKFSeq.Parent = parent

	for i, keyframe in pairs(keyframes) do
		local newKF = Instance.new("Keyframe")
		newKF.Name = tostring(i)

		poseDataToObject("HumanoidRootPart", keyframe.Poses.HumanoidRootPart, newKF)

		newKF.Time = keyframe.Time
		newKF.Parent = newKFSeq
	end
end

return module
